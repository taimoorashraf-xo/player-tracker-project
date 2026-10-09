import os
import tempfile
from collections import Counter, defaultdict

import cv2
import easyocr
import numpy as np
import pandas as pd
import streamlit as st
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# =====================================================
# SETTINGS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Smaller models are much faster on CPU; the weights download on first use.
MODEL_CHOICES = {
    "Nano - fastest": "yolov8n.pt",
    "Small - balanced": "yolov8s.pt",
    "Medium - most accurate": "yolov8m.pt",
}
WIDTH_CHOICES = [480, 640, 960, 1280]   # frames wider than this are scaled down

PERSON_CONF = 0.4           # YOLO confidence threshold for "person"
OCR_EVERY_N_FRAMES = 10     # try reading a shirt number on every Nth frame per track
OCR_MAX_ATTEMPTS = 10       # give up on a track after this many OCR attempts
OCR_MIN_CONF = 0.3

# HSV ranges (OpenCV hue is 0-180). Red wraps around, so it has two ranges.
TEAM_RANGES = {
    "Red": [((0, 100, 70), (10, 255, 255)), ((170, 100, 70), (180, 255, 255))],
    "Blue": [((100, 100, 50), (130, 255, 255))],
    "Yellow": [((18, 100, 100), (35, 255, 255))],
    "Green": [((40, 80, 50), (85, 255, 255))],
    "White": [((0, 0, 170), (180, 50, 255))],
    "Black": [((0, 0, 0), (180, 255, 60))],
}
TEAM_MIN_SHARE = 0.2        # a team colour must cover at least 20% of the shirt area
TEAM_BOX_COLORS = {         # BGR colours used for drawing boxes
    "Red": (0, 0, 255),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
    "Green": (0, 255, 0),
    "White": (230, 230, 230),
    "Black": (60, 60, 60),
    "Unknown": (0, 255, 0),
}


def _majority(counter):
    """Most common key in a Counter, or None if empty."""
    return counter.most_common(1)[0][0] if counter else None


def _new_track_state():
    return {
        "frames": 0,
        "teams": Counter(),
        "numbers": Counter(),
        "ocr_attempts": 0,
    }


def shirt_area(crop):
    """Torso region of a player crop, where the shirt colour and number are."""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    torso = crop[int(0.15 * h):int(0.55 * h), int(0.2 * w):int(0.8 * w)]
    return torso if torso.size else None


def draw_players(frame, players):
    """Return a copy of frame with a labelled box for each player."""
    annotated = frame.copy()
    for track_id, info in players.items():
        x1, y1, x2, y2 = info["bbox"]
        color = TEAM_BOX_COLORS.get(info["team"], TEAM_BOX_COLORS["Unknown"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{track_id} | #{info['shirt_number']} | {info['team']}"
        cv2.putText(
            annotated, label, (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )
    return annotated


# =====================================================
# PLAYER TRACKER CLASS
# =====================================================
class PlayerTracker:

    def __init__(self, model, ocr):
        self.model = model
        self.ocr = ocr          # may be None, in which case shirt numbers are skipped
        self.reset()

    def reset(self):
        """Clear all tracking state so a new video starts from scratch."""
        self.deepsort = DeepSort(embedder="mobilenet", max_age=30, n_init=2, nn_budget=100)
        self.tracks = defaultdict(_new_track_state)
        self.frames_processed = 0

    def detect_team(self, shirt):
        """Return the dominant team colour in the shirt area, or 'Unknown'."""
        if shirt is None:
            return "Unknown"

        hsv = cv2.cvtColor(shirt, cv2.COLOR_BGR2HSV)
        total = shirt.shape[0] * shirt.shape[1]

        best_team, best_share = "Unknown", 0.0
        for team, ranges in TEAM_RANGES.items():
            pixels = 0
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
                pixels += np.count_nonzero(mask)
            share = pixels / total
            if share > best_share:
                best_team, best_share = team, share

        return best_team if best_share >= TEAM_MIN_SHARE else "Unknown"

    def read_shirt_number(self, shirt):
        """Read a 1-2 digit shirt number from the shirt area with OCR."""
        if self.ocr is None or shirt is None:
            return None

        results = self.ocr.readtext(shirt, allowlist="0123456789")
        best_text, best_conf = None, 0.0
        for _, text, conf in results:
            if text.isdigit() and len(text) <= 2 and int(text) <= 99 and conf >= OCR_MIN_CONF:
                if conf > best_conf:
                    best_text, best_conf = text, conf

        return str(int(best_text)).zfill(2) if best_text else None

    def process_frame(self, frame):
        """Detect and track players in one frame. Returns (annotated_frame, players)."""
        self.frames_processed += 1

        results = self.model(frame, classes=[0], conf=PERSON_CONF, verbose=False)
        boxes = results[0].boxes

        detection_list = []
        if boxes is not None and len(boxes) > 0:
            for (x1, y1, x2, y2), conf in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
                detection_list.append(
                    ([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(conf), "player")
                )

        tracks = self.deepsort.update_tracks(detection_list, frame=frame)

        players = {}
        frame_h, frame_w = frame.shape[:2]

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = (int(v) for v in track.to_ltrb())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            shirt = shirt_area(frame[y1:y2, x1:x2])
            state = self.tracks[track_id]
            state["frames"] += 1

            team = self.detect_team(shirt)
            if team != "Unknown":
                state["teams"][team] += 1

            # OCR is slow, so only try on some frames, and stop after a few attempts
            due = (state["frames"] - 1) % OCR_EVERY_N_FRAMES == 0
            if due and state["ocr_attempts"] < OCR_MAX_ATTEMPTS:
                state["ocr_attempts"] += 1
                number = self.read_shirt_number(shirt)
                if number:
                    state["numbers"][number] += 1

            players[track_id] = {
                "bbox": (x1, y1, x2, y2),
                "team": _majority(state["teams"]) or "Unknown",
                "shirt_number": _majority(state["numbers"]) or "?",
            }

        return draw_players(frame, players), players

    def get_stats(self):
        """One row per tracked player, sorted by how often they were seen."""
        total = max(1, self.frames_processed)
        rows = []
        for track_id, state in self.tracks.items():
            if state["frames"] == 0:
                continue
            rows.append({
                "Player ID": track_id,
                "Shirt Number": _majority(state["numbers"]) or "N/A",
                "Team": _majority(state["teams"]) or "Unknown",
                "Frames Detected": state["frames"],
                "Detection Rate": f"{state['frames'] / total * 100:.1f}%",
            })

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("Frames Detected", ascending=False)


# =====================================================
# CACHED MODEL LOADERS
# =====================================================
@st.cache_resource(show_spinner=False)
def load_yolo(model_name):
    # Missing weights are downloaded next to this script on first use
    return YOLO(os.path.join(BASE_DIR, model_name))


@st.cache_resource(show_spinner=False)
def load_ocr():
    """Returns (reader, error_message). The reader is None if EasyOCR failed to load."""
    try:
        return easyocr.Reader(["en"], gpu=False), None
    except Exception as e:
        return None, f"Shirt-number OCR is disabled because EasyOCR failed to load: {e}"


# =====================================================
# VIDEO PROCESSING
# =====================================================
def open_video_writer(path, fps):
    """H.264 writer that browsers can play. Returns None if the encoder is unavailable."""
    try:
        import imageio
        return imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1)
    except Exception as e:
        st.warning(f"Could not create the output video ({e}). Statistics will still be produced.")
        return None


def process_video(tracker, video_file, stride, max_width, show_preview):
    """Run the tracker over an uploaded video.

    Returns {"stats": DataFrame, "video": mp4 bytes or None}, or None on failure.
    Every source frame is written to the output video; frames that are skipped
    for speed reuse the most recent boxes so the result plays back smoothly.
    """
    tracker.reset()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name
    out_path = video_path + ".out.mp4"
    writer = None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("❌ Could not open video file")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30
        st.info(f"📊 Frames: {total_frames} | FPS: {fps:.1f} | Analysing every {stride} frame(s)")

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        placeholder = st.empty()

        players = {}
        frame_no = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1

            h, w = frame.shape[:2]
            if w > max_width:
                frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
            h, w = frame.shape[:2]
            frame = frame[:h - h % 2, :w - w % 2]    # H.264 needs even dimensions

            if (frame_no - 1) % stride == 0:
                annotated, players = tracker.process_frame(frame)
                if show_preview:
                    placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
            else:
                annotated = draw_players(frame, players)

            if writer is None and frame_no == 1:
                writer = open_video_writer(out_path, fps)
            if writer is not None:
                writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            if frame_no % 5 == 0 or frame_no == total_frames:
                if total_frames > 0:
                    progress_bar.progress(min(frame_no / total_frames, 1.0))
                    status_text.text(f"Frame {frame_no}/{total_frames} | Players on screen: {len(players)}")
                else:
                    status_text.text(f"Frame {frame_no} | Players on screen: {len(players)}")

        cap.release()
        if writer is not None:
            writer.close()
            writer = None

        progress_bar.progress(1.0)
        status_text.text(f"Done. Analysed {tracker.frames_processed} of {frame_no} frames.")

        video_bytes = None
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            with open(out_path, "rb") as f:
                video_bytes = f.read()

        return {"stats": tracker.get_stats(), "video": video_bytes}

    except Exception as e:
        st.error(f"❌ Processing error: {e}")
        return None

    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        for path in (video_path, out_path):
            try:
                os.unlink(path)
            except OSError:
                pass


# =====================================================
# MAIN APP
# =====================================================
def main():
    st.set_page_config(page_title="⚽ Player Tracker", layout="wide")

    st.title("⚽ Player Tracking System")
    st.write("Upload → Track → Analyze")
    st.caption(f"Running on: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU - use the Nano model for speed'}")

    video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv", "flv"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        model_label = st.selectbox(
            "Detection model", list(MODEL_CHOICES), index=0,
            help="Nano is the fastest. Medium is the most accurate but slow on CPU.",
        )
    with c2:
        max_width = st.selectbox(
            "Max video width (px)", WIDTH_CHOICES, index=1,
            help="Larger videos are scaled down to this width. Smaller is faster.",
        )
    with c3:
        stride = st.number_input(
            "Analyse every N frames", min_value=1, max_value=30, value=3, step=1,
            help="1 analyses every frame (slowest, most precise). Skipped frames reuse the latest boxes.",
        )
    with c4:
        show_preview = st.checkbox(
            "Show live preview", value=False,
            help="Shows frames while processing. This slows processing down; the finished video plays smoothly.",
        )

    process_btn = st.button("Process", type="primary", disabled=video_file is None)

    if video_file is not None and process_btn:
        try:
            with st.spinner("Loading models (the first use of a model downloads it)..."):
                model = load_yolo(MODEL_CHOICES[model_label])
                ocr, ocr_warning = load_ocr()
        except Exception as e:
            st.error(f"❌ Failed to load models: {e}")
            st.stop()

        if ocr_warning:
            st.warning(ocr_warning)

        tracker = PlayerTracker(model, ocr)
        st.session_state["result"] = process_video(
            tracker, video_file, int(stride), int(max_width), show_preview
        )

    # Results live in session state so that clicking Download does not wipe them
    result = st.session_state.get("result")
    if result is not None:
        if result["video"]:
            st.subheader("🎬 Annotated Video")
            st.video(result["video"])
            st.download_button("📥 Download video", result["video"], "tracked_players.mp4", "video/mp4")

        st.subheader("📊 Player Statistics")
        stats_df = result["stats"]
        if stats_df.empty:
            st.warning("⚠️ No players detected in video")
        else:
            st.dataframe(stats_df, width="stretch", hide_index=True)
            st.download_button(
                "📥 Download CSV",
                stats_df.to_csv(index=False),
                "player_stats.csv",
                "text/csv",
            )


if __name__ == "__main__":
    main()
