import os
import tempfile
from collections import Counter, defaultdict

import cv2
import easyocr
import numpy as np
import pandas as pd
import streamlit as st
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# =====================================================
# SETTINGS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8m.pt")

MAX_WIDTH = 1280            # frames wider than this are scaled down (aspect kept)
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


# =====================================================
# PLAYER TRACKER CLASS
# =====================================================
class PlayerTracker:

    def __init__(self):
        self.warnings = []
        self.model = YOLO(MODEL_PATH)
        self.ocr = self._load_ocr()
        self.reset()

    def _load_ocr(self):
        try:
            return easyocr.Reader(["en"], gpu=False)
        except Exception as e:
            self.warnings.append(
                f"Shirt-number OCR is disabled because EasyOCR failed to load: {e}"
            )
            return None

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
        """Detect, track and annotate one frame. Returns (annotated_frame, players)."""
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

        annotated = frame.copy()
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

            team_now = _majority(state["teams"]) or "Unknown"
            number_now = _majority(state["numbers"]) or "?"

            players[track_id] = {
                "bbox": (x1, y1, x2, y2),
                "team": team_now,
                "shirt_number": number_now,
            }

            color = TEAM_BOX_COLORS.get(team_now, TEAM_BOX_COLORS["Unknown"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id} | #{number_now} | {team_now}"
            cv2.putText(
                annotated, label, (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

        return annotated, players

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
# CACHE
# =====================================================
@st.cache_resource(show_spinner=False)
def load_tracker():
    return PlayerTracker()


# =====================================================
# VIDEO PROCESSING
# =====================================================
def process_video(tracker, video_file, stride):
    """Run the tracker over an uploaded video. Returns the stats DataFrame."""
    tracker.reset()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("❌ Could not open video file")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        st.info(f"📊 Frames: {total_frames} | FPS: {fps:.1f} | Processing every {stride} frame(s)")

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        placeholder = st.empty()

        frame_no = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            if (frame_no - 1) % stride:
                continue

            h, w = frame.shape[:2]
            if w > MAX_WIDTH:
                frame = cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / w)))

            annotated, players = tracker.process_frame(frame)
            placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")

            if total_frames > 0:
                progress = min(frame_no / total_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Frame {frame_no}/{total_frames} | Players on screen: {len(players)}")
            else:
                status_text.text(f"Frame {frame_no} | Players on screen: {len(players)}")

        cap.release()
        progress_bar.progress(1.0)
        status_text.text(f"Done. Processed {tracker.frames_processed} frames.")
        return tracker.get_stats()

    except Exception as e:
        st.error(f"❌ Processing error: {e}")
        return None

    finally:
        try:
            os.unlink(video_path)
        except OSError:
            pass


# =====================================================
# MAIN APP
# =====================================================
def main():
    st.title("⚽ Player Tracking System")
    st.write("Upload → Track → Analyze")

    try:
        with st.spinner("Loading models (the first run downloads OCR weights)..."):
            tracker = load_tracker()
    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        st.stop()

    for message in tracker.warnings:
        st.warning(message)

    col1, col2 = st.columns([3, 1])
    with col1:
        video_file = st.file_uploader(
            "Upload video",
            type=["mp4", "avi", "mov", "mkv", "flv"],
        )
    with col2:
        stride = st.number_input(
            "Process every N frames",
            min_value=1, max_value=30, value=3, step=1,
            help="Higher is faster but less precise. 1 processes every frame.",
        )
        process_btn = st.button("Process", width="stretch")

    if video_file is not None and process_btn:
        st.session_state["stats"] = process_video(tracker, video_file, int(stride))

    # Results live in session state so that clicking Download does not wipe them
    stats_df = st.session_state.get("stats")
    if stats_df is not None:
        st.subheader("📊 Player Statistics")
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
