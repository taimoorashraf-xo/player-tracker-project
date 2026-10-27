"""Finding players in a video, following them, and measuring how they move.

The pipeline for one clip:

1. YOLO finds people in each analysed frame; anyone not standing on grass is dropped.
2. `tracking.CameraModel` works out how the camera moved, ignoring the players themselves,
   and flags cuts to a new shot.
3. Detections are mapped into the shot's reference frame so the tracker compares like with
   like, and Deep SORT follows each player using their kit colour as the appearance cue.
4. Once the clip is done, a perspective scale is fitted per shot, positions become metres,
   and the movement numbers come out of `video_metrics`.
5. Tracks from different shots that look like the same player are merged.
"""
import os
import tempfile
from collections import Counter, defaultdict

import cv2
import numpy as np
import pandas as pd
from deep_sort_realtime.deepsort_tracker import DeepSort

import tracking as tk
import video_metrics as vm

PERSON_CONF = 0.4           # YOLO confidence threshold for "person"
MIN_BOX_HEIGHT_FRAC = 0.035  # ignore detections shorter than this share of the frame
MAX_BOX_HEIGHT_FRAC = 0.85  # and anything implausibly tall for a player
MAX_ASPECT = 1.1            # width / height; wider than this is not one standing person
OCR_EVERY_N_FRAMES = 8
OCR_MAX_ATTEMPTS = 12
OCR_MIN_CONF = 0.35
THUMB_HEIGHT = 150

GRASS_LOWER = (32, 35, 25)
GRASS_UPPER = (92, 255, 255)
PITCH_MIN_GRASS = 0.25      # grass share around the feet needed to count as on the pitch

FOCAL_LAG_S = 0.5           # spacing used when inferring the focal length from movement

TRACKER_SETTINGS = dict(embedder=None, max_age=25, n_init=3, nn_budget=120,
                        max_cosine_distance=0.35, max_iou_distance=0.75)


def grass_mask(hsv):
    return cv2.inRange(hsv, np.array(GRASS_LOWER, np.uint8), np.array(GRASS_UPPER, np.uint8))


def shirt_area(crop):
    """Torso region of a player crop, where the kit colour and number are."""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    torso = crop[int(0.15 * h):int(0.55 * h), int(0.2 * w):int(0.8 * w)]
    return torso if torso.size else None


def on_pitch(grass, box):
    """True if the ground around the person's feet is mostly grass."""
    x1, y1, x2, y2 = box
    height = y2 - y1
    top = max(0, int(y2 - 0.15 * height))
    bottom = min(grass.shape[0], int(y2 + 0.1 * height) + 1)
    band = grass[top:bottom, x1:x2]
    return band.size > 0 and np.count_nonzero(band) / band.size >= PITCH_MIN_GRASS


def _new_track():
    return {
        "shot": 0, "label": None, "frames": 0,
        "teams": Counter(), "numbers": Counter(), "ocr_attempts": 0,
        "obs": [],                  # (t, foot x, foot y, height, edge) in the shot's frame
        "thumb": None, "thumb_h": 0,
        "lab_sum": np.zeros(3), "lab_n": 0,
        "signature": None, "signature_n": 0,
    }


class PlayerTracker:
    """Follows every player through one clip and remembers how each of them moved."""

    def __init__(self, model, ocr=None, pitch_only=True):
        self.model = model
        self.ocr = ocr
        self.pitch_only = pitch_only
        self.reset()

    def reset(self):
        self.deepsort = DeepSort(**TRACKER_SETTINGS)
        self.camera = tk.CameraModel()
        self.tracks = defaultdict(_new_track)
        self.records = []           # per analysed frame: who was where, in image pixels
        self.frames_analysed = 0
        self.shot = 0
        self.labels_used = 0
        self.untrusted_frames = 0

    # ---------- detection ----------
    def detect(self, frame):
        """People on the pitch in this frame, as ((x1, y1, x2, y2), confidence)."""
        frame_h, frame_w = frame.shape[:2]
        result = self.model(frame, classes=[0], conf=PERSON_CONF, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        grass = grass_mask(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)) if self.pitch_only else None

        people = []
        for (x1, y1, x2, y2), conf in zip(result.boxes.xyxy.cpu().numpy(),
                                          result.boxes.conf.cpu().numpy()):
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(frame_w, int(x2)), min(frame_h, int(y2))
            height, width = y2 - y1, x2 - x1
            if width <= 0 or height <= 0:
                continue
            if not MIN_BOX_HEIGHT_FRAC * frame_h <= height <= MAX_BOX_HEIGHT_FRAC * frame_h:
                continue
            if width / height > MAX_ASPECT:         # two players merged, or a sliding tackle
                continue
            if grass is not None and not on_pitch(grass, (x1, y1, x2, y2)):
                continue
            people.append(((x1, y1, x2, y2), float(conf)))
        return people

    def team_colour(self, shirt):
        """Mean Lab colour of the shirt with grass removed, or None if it is mostly grass."""
        if shirt is None:
            return None
        hsv = cv2.cvtColor(shirt, cv2.COLOR_BGR2HSV)
        not_grass = grass_mask(hsv) == 0
        if not_grass.mean() < 0.3:
            return None
        return cv2.cvtColor(shirt, cv2.COLOR_BGR2LAB)[not_grass].mean(axis=0), hsv, not_grass

    def read_number(self, shirt):
        """Read a 1-2 digit shirt number, or None. Only works on close-ups."""
        if self.ocr is None or shirt is None:
            return None
        best, best_conf = None, 0.0
        for _, text, conf in self.ocr.readtext(shirt, allowlist="0123456789"):
            if text.isdigit() and 0 < len(text) <= 2 and int(text) <= 99 and conf >= OCR_MIN_CONF:
                if conf > best_conf:
                    best, best_conf = text, conf
        return str(int(best)).zfill(2) if best else None

    # ---------- per frame ----------
    def process_frame(self, frame, t=0.0, frame_no=0, full_res=None):
        """Detect, follow and measure everyone in one frame. Returns {key: player}."""
        self.frames_analysed += 1
        frame_h, frame_w = frame.shape[:2]
        photo = frame if full_res is None else full_res
        photo_scale = photo.shape[1] / frame_w

        people = self.detect(frame)
        # The camera estimate must ignore the players, or it follows them instead of the pitch
        matrix, cut = self.camera.update(frame, [box for box, _ in people])
        if cut:
            self.shot += 1
            self.deepsort = DeepSort(**TRACKER_SETTINGS)
        if not self.camera.trusted:
            self.untrusted_frames += 1
        zoom = self.camera.scale
        inverse = cv2.invertAffineTransform(matrix)

        detections, embeds, shirts = [], [], []
        for (x1, y1, x2, y2), conf in people:
            centre = tk.apply_affine(matrix, [[(x1 + x2) / 2, (y1 + y2) / 2]])[0]
            width, height = (x2 - x1) * zoom, (y2 - y1) * zoom
            detections.append(([centre[0] - width / 2, centre[1] - height / 2, width, height],
                               conf, "player"))
            shirt = shirt_area(frame[y1:y2, x1:x2])
            shirts.append(shirt)
            embeds.append(self._embedding(shirt))

        tracks = self.deepsort.update_tracks(detections, embeds=embeds) if detections else []

        players = {}
        for track in tracks:
            # A track with no detection this frame is only a prediction: drawing it would
            # put an empty box on the pitch where nobody is.
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            left, top, right, bottom = track.to_ltrb()
            ref_centre = np.array([(left + right) / 2, (top + bottom) / 2])
            ref_h, ref_w = bottom - top, right - left

            img_centre = tk.apply_affine(inverse, [ref_centre])[0]
            img_h, img_w = ref_h / zoom, ref_w / zoom
            x1 = int(img_centre[0] - img_w / 2)
            y1 = int(img_centre[1] - img_h / 2)
            x2, y2 = int(x1 + img_w), int(y1 + img_h)
            edge = x1 < 2 or y1 < 2 or x2 > frame_w - 2 or y2 > frame_h - 2
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            key = f"{self.shot}:{track.track_id}"
            state = self.tracks[key]
            if state["label"] is None:
                self.labels_used += 1
                state["label"] = f"P{self.labels_used}"
                state["shot"] = self.shot
            state["frames"] += 1
            # Stored in the shot's reference frame, where the perspective fit happens
            state["obs"].append((t, float(ref_centre[0]), float(ref_centre[1] + ref_h / 2),
                                 float(ref_h), edge))

            shirt = shirt_area(frame[y1:y2, x1:x2])
            self._absorb_appearance(state, shirt)
            self._absorb_photo(state, photo, photo_scale, (x1, y1, x2, y2), edge)

            players[key] = {
                "bbox": (x1, y1, x2, y2),
                "label": state["label"],
                "team": "Unknown",
                "number": self._top(state["numbers"]) or "",
                "pos": (float(ref_centre[0]), float(ref_centre[1] + ref_h / 2)),
                "edge": edge,
                "shot": self.shot,
            }

        self.records.append({"t": t, "frame_no": frame_no, "shot": self.shot, "players": players})
        return players

    # ---------- appearance ----------
    @staticmethod
    def _embedding(shirt):
        """Kit-colour histogram, used by the tracker to tell two players apart."""
        size = 12 * 4
        flat = np.full(size, 1.0 / np.sqrt(size), np.float32)
        if shirt is None:
            return flat
        hsv = cv2.cvtColor(shirt, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
        norm = np.linalg.norm(hist)
        return (hist / norm).astype(np.float32) if norm else flat

    def _absorb_appearance(self, state, shirt):
        colour = self.team_colour(shirt)
        if colour is None:
            return
        lab_mean, hsv, _ = colour
        state["lab_sum"] += lab_mean
        state["lab_n"] += 1
        signature = tk.colour_signature(hsv)
        if state["signature"] is None:
            state["signature"] = signature
        else:                                   # running average keeps it stable
            n = state["signature_n"]
            state["signature"] = (state["signature"] * n + signature) / (n + 1)
        state["signature_n"] += 1

        due = (state["frames"] - 1) % OCR_EVERY_N_FRAMES == 0
        if due and state["ocr_attempts"] < OCR_MAX_ATTEMPTS:
            state["ocr_attempts"] += 1
            number = self.read_number(shirt)
            if number:
                state["numbers"][number] += 1

    @staticmethod
    def _absorb_photo(state, photo, photo_scale, box, edge):
        """Keep the biggest fully visible view of the player as their picture."""
        x1, y1, x2, y2 = box
        if edge or (y2 - y1) <= state["thumb_h"]:
            return
        crop = photo[int(y1 * photo_scale):int(y2 * photo_scale),
                     int(x1 * photo_scale):int(x2 * photo_scale)]
        if not crop.size:
            return
        ratio = THUMB_HEIGHT / crop.shape[0]
        state["thumb"] = cv2.resize(crop, None, fx=ratio, fy=ratio,
                                    interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC)
        state["thumb_h"] = y2 - y1

    @staticmethod
    def _top(counter):
        return counter.most_common(1)[0][0] if counter else None


# =====================================================
# TURNING TRACKS INTO PLAYERS
# =====================================================
def _median_step(observations):
    """Typical seconds between consecutive observations of one track."""
    gaps = np.diff([o[0] for o in observations]) if len(observations) > 1 else []
    gaps = [g for g in gaps if 0 < g <= vm.MAX_GAP_S]
    return float(np.median(gaps)) if gaps else 0.1


def _scale_models(tracks, frame_width):
    """Fit a perspective scale per camera shot, so each shot's own framing is respected."""
    rows, heights, steps = defaultdict(list), defaultdict(list), defaultdict(list)
    for state in tracks.values():
        shot = state["shot"]
        clean = [o for o in state["obs"] if not o[4]]
        for _, _, foot_y, height, _ in clean:
            rows[shot].append(foot_y)
            heights[shot].append(height)
        # Compare observations about half a second apart: over a single frame the real
        # movement is swamped by box jitter, which would bias the focal length downwards.
        lag = max(1, int(round(FOCAL_LAG_S / _median_step(clean))))
        for (t0, x0, y0, _, _), (t1, x1, y1, _, _) in zip(clean, clean[lag:]):
            if 0 < t1 - t0 <= lag * vm.MAX_GAP_S:
                steps[shot].append((x0, y0, x1, y1))

    models = {}
    for shot in rows:
        model = tk.ScaleModel.fit(rows[shot], heights[shot], frame_width=frame_width)
        model.calibrate_focal(steps[shot], frame_width, frame_width / 2)
        models[shot] = model
    return models


def build_players(tracker, frame_width):
    """Collapse raw tracks into one row per player, with their movement measured.

    Returns {"players": DataFrame, "paths": {player: path}, "swatches": {team: colour},
    "scales": {shot: ScaleModel}, "tracks_before_merge": int}.
    """
    tracks = dict(tracker.tracks)
    scales = _scale_models(tracks, frame_width)
    centre_x = frame_width / 2

    colours = {k: (s["lab_sum"] / s["lab_n"], s["lab_n"]) for k, s in tracks.items() if s["lab_n"] >= 3}
    team_of, swatches = vm.cluster_teams(colours)
    for key, state in tracks.items():
        state["team"] = team_of.get(key)
        state["number"] = PlayerTracker._top(state["numbers"])

    # Pressing needs everyone's position at once, so it is measured shot by shot
    press = {}
    by_shot = defaultdict(list)
    for record in tracker.records:
        by_shot[record["shot"]].append(record)
    for shot, records in by_shot.items():
        if shot in scales:
            press.update(vm.press_metrics(records, team_of, scales[shot], centre_x))

    measured = {}
    for key, state in tracks.items():
        if state["shot"] not in scales:
            continue
        path = vm.ground_track(state["obs"], scales[state["shot"]], centre_x)
        metrics = vm.movement_metrics(path)
        if metrics is None:
            continue
        measured[key] = {"metrics": metrics, "path": path, "state": state}

    groups = tk.merge_across_shots({k: tracks[k] for k in measured})
    merged = defaultdict(list)
    for key, group in groups.items():
        merged[group].append(key)

    rows, paths = [], {}
    for members in merged.values():
        members.sort(key=lambda k: -measured[k]["metrics"]["observed_s"])
        main = measured[members[0]]["state"]
        weights = np.array([measured[k]["metrics"]["observed_s"] for k in members], dtype=float)
        combined = {}
        for field in ("top_speed_kmh", "avg_speed_kmh", "hi_share", "distance_m_per_min",
                      "sprints_per_min", "accels_per_min"):
            values = np.array([measured[k]["metrics"][field] for k in members], dtype=float)
            # Top speed is the best moment anywhere; the rest average over time on screen
            combined[field] = float(values.max() if field == "top_speed_kmh"
                                    else np.average(values, weights=weights))
        press_values = [press[k]["press_share"] for k in members if k in press]
        combined["press_share"] = float(np.mean(press_values)) if press_values else np.nan

        numbers = Counter()
        for key in members:
            numbers.update(tracks[key]["numbers"])
        label = main["label"]
        rows.append({
            "key": label, "player": label, "tracks": len(members),
            "team": main["team"] or "Unknown",
            "shirt": PlayerTracker._top(numbers) or "",
            "observed_s": float(weights.sum()),
            "distance_m": float(sum(measured[k]["metrics"]["distance_m"] for k in members)),
            "thumb": max((measured[k]["state"] for k in members), key=lambda s: s["thumb_h"])["thumb"],
            "track_keys": members,
            **combined,
        })
        paths[label] = measured[members[0]]["path"]

    players = pd.DataFrame(rows)
    if not players.empty:
        players = players.sort_values("observed_s", ascending=False).reset_index(drop=True)
    return {"players": players, "paths": paths, "swatches": swatches, "scales": scales,
            "tracks_before_merge": len(measured), "team_of": team_of}


# =====================================================
# DRAWING
# =====================================================
FONT = cv2.FONT_HERSHEY_DUPLEX
HIGHLIGHT = (60, 200, 255)          # BGR amber for the picked players
MUTED = (150, 150, 150)


def draw_frame(frame, players, highlights=None, team_colours=None):
    """Thin boxes for everyone; bold amber boxes and a tag for the highlighted players."""
    out = frame.copy()
    highlights = highlights or {}
    scale = max(0.4, frame.shape[0] / 1100)
    thick = max(2, frame.shape[0] // 320)

    for key, info in players.items():
        if key in highlights:
            continue
        x1, y1, x2, y2 = info["bbox"]
        colour = (team_colours or {}).get(info.get("team"), MUTED)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 1)

    for key, info in players.items():
        if key not in highlights:
            continue
        rank, score = highlights[key]
        x1, y1, x2, y2 = info["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), HIGHLIGHT, thick)
        label = f"{rank}  {info['label']}"
        if info.get("number"):
            label += f"  #{info['number']}"
        label += f"   {score:.0f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, scale, 1)
        top = max(th + 8, y1)
        cv2.rectangle(out, (x1, top - th - 8), (x1 + tw + 10, top), HIGHLIGHT, -1)
        cv2.putText(out, label, (x1 + 5, top - 5), FONT, scale, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def fit_frame(frame, max_width):
    """Scale down to max_width and trim to even dimensions, which H.264 requires."""
    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
    return frame[:h - h % 2, :w - w % 2]


def open_writer(path, fps):
    """H.264 writer a browser can play, or None if the encoder is missing."""
    try:
        import imageio
        return imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1,
                                  ffmpeg_params=["-crf", "23", "-preset", "veryfast"])
    except Exception:
        return None


# =====================================================
# WHOLE-CLIP PIPELINE
# =====================================================
def analyse_clip(tracker, video_bytes, stride, max_width, on_progress=None, on_frame=None):
    """Run the tracker over a clip. Returns the analysis, or raises on an unreadable file."""
    tracker.reset()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        path = tmp.name
    try:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise ValueError("That file could not be opened as a video.")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0:
            fps = 30.0

        frame_no, width = 0, None
        while True:
            ok, original = capture.read()
            if not ok:
                break
            frame_no += 1
            if (frame_no - 1) % stride:
                continue
            frame = fit_frame(original, max_width)
            width = frame.shape[1]
            players = tracker.process_frame(frame, t=(frame_no - 1) / fps,
                                            frame_no=frame_no, full_res=original)
            if on_frame:
                on_frame(frame, players)
            if on_progress and total > 0:
                on_progress(min(frame_no / total, 1.0), frame_no, total, len(players))
        capture.release()

        result = build_players(tracker, width or max_width)
        result.update({"fps": fps, "frames": frame_no, "analysed": tracker.frames_analysed,
                       "shots": tracker.shot + 1, "max_width": max_width, "video": video_bytes,
                       "records": tracker.records,
                       "untrusted_share": tracker.untrusted_frames / max(tracker.frames_analysed, 1)})
        return result
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def render_highlight_video(analysis, highlights, team_colours=None):
    """Re-encode the clip with the chosen players boxed. Returns mp4 bytes, or None."""
    boxes_at = {r["frame_no"]: r["players"] for r in analysis["records"]}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(analysis["video"])
        path = tmp.name
    out_path = path + ".out.mp4"
    writer = None
    try:
        capture = cv2.VideoCapture(path)
        players, frame_no = {}, 0
        while True:
            ok, original = capture.read()
            if not ok:
                break
            frame_no += 1
            frame = fit_frame(original, analysis["max_width"])
            players = boxes_at.get(frame_no, players)
            if writer is None:
                writer = open_writer(out_path, analysis["fps"])
                if writer is None:
                    return None
            drawn = draw_frame(frame, players, highlights, team_colours)
            writer.append_data(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
        capture.release()
        if writer is not None:
            writer.close()
            writer = None
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            with open(out_path, "rb") as handle:
                return handle.read()
        return None
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        for p in (path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass
