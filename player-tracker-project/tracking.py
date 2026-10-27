"""Camera-compensated player tracking and ground-plane motion estimation.

Broadcast footage pans, tilts and zooms. Measuring a player's speed straight from pixel
movement therefore measures the camera as much as the player. This module removes the
camera motion and converts pixel movement into approximate metres on the pitch:

1. `CameraModel` estimates a per-frame similarity transform (zoom, rotation, pan) from
   background features, masking out the players so their movement cannot corrupt it.
   Phase correlation, which only estimates a shift, cannot see zoom at all.
2. `ScaleModel` fits how many pixels represent a metre at each image row, using the
   apparent height of every player seen in the clip. Perspective makes distant players
   smaller, so one scale for the whole frame is wrong.
3. `ground_xy` turns a stabilised image point into metres on the pitch, integrating the
   scale down the image so movement towards or away from the camera counts properly.

All of it is an approximation: it assumes the pitch is flat and the camera rotates about
roughly one point, which holds well enough for a main broadcast camera and poorly for
close-ups and Steadicam shots.
"""
import numpy as np

try:                                    # OpenCV is optional only so the maths can be unit-tested
    import cv2
except ImportError:                     # pragma: no cover
    cv2 = None

PLAYER_HEIGHT_M = 1.80                  # assumed standing height, the only real-world anchor
ANALYSIS_WIDTH = 480                    # camera motion is estimated on a frame this wide
MAX_CORNERS = 600
MIN_MATCHES = 5                         # fewer tracked features than this: nothing to estimate from
MIN_INLIERS = 8                         # fewer background matches than this: estimate not fully trusted
CUT_INLIER_RATIO = 0.35                 # below this share of matching features, the move is lost
CUT_SCALE_JUMP = 0.25                   # a >25% zoom change in one frame is a cut, not a zoom
CUT_SIMILARITY = 0.45                   # layouts less alike than this are a different shot
SIGNATURE_GRID = 6                      # the picture is summarised as this many blocks a side
MAX_BRIDGE_FRAMES = 2                   # brief feature loss is carried over, not treated as a cut
SCALE_BINS = 8                          # image rows are binned this many ways for the scale fit
MIN_SCALE_SAMPLES = 25
MIN_FOCAL_SAMPLES = 60
DEFAULT_FOV_DEGREES = 35.0      # assumed horizontal field of view of a broadcast camera
FOCAL_ADJUST_RANGE = (0.65, 1.5)  # how far movement in the clip may move that assumption
PPM_FLOOR = 1e-3


# =====================================================
# CAMERA MOTION
# =====================================================
def focal_from_fov(frame_width, fov_degrees=DEFAULT_FOV_DEGREES):
    """Focal length in pixels for an assumed horizontal field of view."""
    return float(frame_width / 2 / np.tan(np.radians(fov_degrees) / 2))


def frame_signature(gray):
    """Coarse layout of the picture, used to tell a cut from a tricky camera move.

    A grid of block brightnesses, centred and normalised so it can be compared by dot
    product. Overall brightness is useless for this: two shots of the same pitch have
    near-identical colour histograms, while their layouts are completely different.
    """
    blocks = cv2.resize(gray, (SIGNATURE_GRID, SIGNATURE_GRID),
                        interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    blocks -= blocks.mean()
    norm = np.linalg.norm(blocks)
    return blocks / norm if norm else blocks


def _similarity_scale(matrix):
    """Zoom factor of a 2x3 affine matrix."""
    return float(np.sqrt(abs(np.linalg.det(matrix[:, :2]))))


def compose(outer, inner):
    """Combine two 2x3 affine transforms so that compose(A, B) applies B then A."""
    a = np.vstack([outer, [0, 0, 1]])
    b = np.vstack([inner, [0, 0, 1]])
    return (a @ b)[:2]


def apply_affine(matrix, points):
    """Apply a 2x3 transform to an (N, 2) array of points."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    return pts @ matrix[:, :2].T + matrix[:, 2]


class CameraModel:
    """Tracks how the camera has moved since the first frame of the current shot.

    `update` returns (transform, cut). The transform maps a point in the current frame to
    the equivalent point in the shot's first frame, so player positions can be compared
    across frames. `cut` is True when the picture changed too much to be the same shot.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.to_reference = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.prev_gray = None
        self.prev_points = None
        self.scale = 1.0                # zoom relative to the first frame of the shot
        self.last_inliers = 0
        self.trusted = True             # whether THIS frame's estimate rests on enough features
        self.prev_signature = None
        self.last_step = None           # the most recent frame-to-frame move, used to bridge gaps
        self.missed = 0                 # consecutive frames the features could not be matched

    @staticmethod
    def _background_mask(shape, boxes, shrink):
        """White everywhere except inside the player boxes."""
        mask = np.full(shape, 255, np.uint8)
        for x1, y1, x2, y2 in boxes:
            pad_x = int(0.15 * (x2 - x1))
            pad_y = int(0.15 * (y2 - y1))
            mask[max(0, int((y1 - pad_y) * shrink)):int((y2 + pad_y) * shrink),
                 max(0, int((x1 - pad_x) * shrink)):int((x2 + pad_x) * shrink)] = 0
        return mask

    def update(self, frame, player_boxes=()):
        """Estimate this frame's transform. Returns (2x3 transform to reference, cut)."""
        h, w = frame.shape[:2]
        shrink = ANALYSIS_WIDTH / w if w > ANALYSIS_WIDTH else 1.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if shrink != 1.0:
            gray = cv2.resize(gray, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
        mask = self._background_mask(gray.shape, player_boxes, shrink)

        signature = frame_signature(gray)
        looks_different = (self.prev_signature is not None
                           and float(self.prev_signature @ signature) < CUT_SIMILARITY)

        step, lost = None, False
        if self.prev_gray is not None and self.prev_points is not None and len(self.prev_points) >= MIN_MATCHES:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            ok = (status.ravel() == 1)
            src = self.prev_points.reshape(-1, 2)[ok]
            dst = nxt.reshape(-1, 2)[ok]
            # Keep only features that are still on the background in the new frame
            if len(dst):
                cols = np.clip(dst[:, 0].astype(int), 0, gray.shape[1] - 1)
                rows = np.clip(dst[:, 1].astype(int), 0, gray.shape[0] - 1)
                on_background = mask[rows, cols] > 0
                src, dst = src[on_background], dst[on_background]

            if len(src) >= MIN_MATCHES:
                # Estimate current -> previous, so transforms chain back to the reference frame
                matrix, inliers = cv2.estimateAffinePartial2D(
                    dst, src, method=cv2.RANSAC, ransacReprojThreshold=2.5,
                    maxIters=2000, confidence=0.995)
                n_in = int(inliers.sum()) if inliers is not None else 0
                self.last_inliers = n_in
                ratio = n_in / len(src)
                zoom = _similarity_scale(matrix) if matrix is not None else 1.0
                if matrix is None or n_in < MIN_MATCHES or ratio < CUT_INLIER_RATIO \
                        or abs(zoom - 1.0) > CUT_SCALE_JUMP:
                    lost = True
                else:
                    step = matrix.copy()
                    step[:, 2] /= shrink              # back to full-resolution pixels
                    self.trusted = n_in >= MIN_INLIERS
            else:
                lost = True
        self.prev_signature = signature

        # Losing the features is not the same as a cut. Mown grass repeats, so a pan can
        # briefly defeat the matcher while the picture stays the same shot; a real cut
        # also rearranges the whole picture. Short losses are bridged by carrying the last
        # move forward, which beats splitting a player's run in two.
        cut = False
        if lost:
            self.missed += 1
            if looks_different or self.missed > MAX_BRIDGE_FRAMES:
                cut = True
            elif self.last_step is not None:
                step = self.last_step
                self.trusted = False
        else:
            self.missed = 0
            if step is not None:
                self.last_step = step

        if cut:
            # A new shot has its own reference frame; nothing before it can be compared
            self.to_reference = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            self.scale, self.trusted = 1.0, True
            self.last_step, self.missed = None, 0
        elif step is not None:
            self.to_reference = compose(self.to_reference, step)
            self.scale = _similarity_scale(self.to_reference)
        elif self.prev_gray is not None:
            self.trusted = False        # kept the old transform; positions may drift

        self.prev_gray = gray
        self.prev_points = cv2.goodFeaturesToTrack(
            gray, maxCorners=MAX_CORNERS, qualityLevel=0.004, minDistance=6, mask=mask, blockSize=7)
        return self.to_reference, cut


# =====================================================
# PERSPECTIVE SCALE
# =====================================================
class ScaleModel:
    """How image pixels relate to metres on the pitch, and how that changes with depth.

    For a camera looking across a flat pitch, a player's apparent height in pixels is
    `ppm(row) = (row - horizon) / camera_height`: a straight line down the image. Fitting
    that line over every player seen gives a far steadier scale than each player's own box,
    which jumps when they are partly hidden or bending.

    Sideways and depth movement do not share a scale. One metre sideways always shifts the
    image by `ppm` pixels, but one metre away from the camera shifts it by less and less the
    further away the player is. Depth therefore uses `depth = focal / ppm`, which is the
    real pinhole relation; the old code integrated `1 / ppm`, which squashed every run
    towards or away from the camera.
    """

    def __init__(self, a=1.0, b=0.0, focal=None):
        self.a, self.b = float(a), float(b)
        self.focal = focal              # pixels; sets how much depth movement counts
        self.samples = 0
        self.fitted = False

    @classmethod
    def fit(cls, rows, heights, frame_width=None):
        """Fit from foot rows and box heights (both in stabilised pixels)."""
        rows = np.asarray(rows, dtype=float)
        ppm = np.asarray(heights, dtype=float) / PLAYER_HEIGHT_M
        keep = np.isfinite(rows) & np.isfinite(ppm) & (ppm > PPM_FLOOR)
        rows, ppm = rows[keep], ppm[keep]
        default_focal = focal_from_fov(frame_width) if frame_width else None

        if len(rows) < MIN_SCALE_SAMPLES:
            model = cls(float(np.median(ppm)) if len(ppm) else 1.0, 0.0, default_focal)
            model.samples = len(rows)
            return model

        # Median height per row band resists players who are crouching, jumping or clipped
        edges = np.unique(np.quantile(rows, np.linspace(0, 1, SCALE_BINS + 1)))
        centres, medians, weights = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            band = (rows >= lo) & (rows <= hi)
            if band.sum() >= 3:
                centres.append(rows[band].mean())
                medians.append(np.median(ppm[band]))
                weights.append(band.sum())

        model = cls(float(np.median(ppm)), 0.0, default_focal)
        model.samples = len(rows)
        if len(centres) >= 3:
            b, a = np.polyfit(centres, medians, 1, w=np.sqrt(weights))
            span = np.array([rows.min(), rows.max()])
            if b > PPM_FLOOR and np.all(a + b * span > PPM_FLOOR):
                model.a, model.b, model.fitted = float(a), float(b), True
        return model

    def calibrate_focal(self, steps, frame_width, centre_x, fov_degrees=DEFAULT_FOV_DEGREES):
        """Settle how much a change in apparent size counts as distance.

        Player heights fix the scale across the picture but say nothing about depth: that
        needs the camera's field of view, which no amount of watching players can reveal.
        The starting point is therefore an assumption about broadcast framing, and the only
        thing measured here is a bounded correction to it.

        The correction comes from movement. Over many players the sideways and depth parts
        of running are of broadly similar size, so comparing them suggests a focal length.
        That comparison is biased, because players running towards or away from the camera
        stay in shot longer than players crossing it, so it is allowed to move the
        assumption by a limited amount rather than replace it.

        Each step is a pair of stabilised positions (x0, y0, x1, y1) about half a second
        apart. Note that sideways is not simply the image's x: a player moving away off to
        one side also slides across the picture, so the sideways distance is taken from the
        full ground mapping.
        """
        default = focal_from_fov(frame_width, fov_degrees)
        low, high = FOCAL_ADJUST_RANGE[0] * default, FOCAL_ADJUST_RANGE[1] * default
        self.focal = default
        self.focal_assumed = True
        if not self.fitted or len(steps) < MIN_FOCAL_SAMPLES:
            return self.focal

        x0, y0, x1, y1 = (np.asarray(c, dtype=float) for c in zip(*steps))
        inv0, inv1 = 1.0 / self.ppm(y0), 1.0 / self.ppm(y1)
        across = (x1 - centre_x) * inv1 - (x0 - centre_x) * inv0     # metres across the pitch
        into_per_focal = inv1 - inv0                                 # depth, before the focal length
        usable = np.isfinite(across) & np.isfinite(into_per_focal)
        if usable.sum() >= MIN_FOCAL_SAMPLES:
            # Medians, not means: a few tracking glitches must not set the scale
            spread_across = np.median(across[usable] ** 2)
            spread_into = np.median(into_per_focal[usable] ** 2)
            if spread_into > 0:
                self.focal = float(np.clip(np.sqrt(spread_across / spread_into), low, high))
                self.focal_assumed = False
        return self.focal

    def ppm(self, row):
        """Pixels per metre at an image row."""
        return np.maximum(self.a + self.b * np.asarray(row, dtype=float), PPM_FLOOR)

    def ground_xy(self, x, y, centre_x=0.0):
        """Stabilised image point -> metres (across the pitch, away from the camera).

        Only differences between points are meaningful; the origin is arbitrary.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ppm_here = self.ppm(y)
        across = (x - centre_x) / ppm_here
        if self.b > PPM_FLOOR and self.focal:
            depth = self.focal / ppm_here          # pinhole: distance is inverse to apparent size
        else:
            depth = y / max(self.a, PPM_FLOOR)     # no perspective detected: treat as flat
        return across, depth


# =====================================================
# RE-IDENTIFICATION ACROSS CAMERA CUTS
# =====================================================
def colour_signature(hsv_crop):
    """Normalised hue/saturation histogram of a shirt, used to recognise a kit again."""
    hist = cv2.calcHist([hsv_crop], [0, 1], None, [16, 6], [0, 180, 0, 256]).flatten()
    total = hist.sum()
    return (hist / total).astype(np.float32) if total > 0 else hist.astype(np.float32)


def _similar(one, other):
    """1.0 for identical colour signatures, 0.0 for nothing in common."""
    if one is None or other is None or one.size != other.size:
        return 0.0
    return float(np.minimum(one, other).sum())        # histogram intersection


def merge_across_shots(tracks, min_similarity=0.5):
    """Group tracks that are the same player seen in different camera shots.

    Within one shot the tracker already holds a player's identity, so only tracks from
    different shots are considered. Teammates wear the same kit, so kit colour alone cannot
    tell them apart across a cut; a **matching shirt number** is required, with the kit
    colour as a sanity check. Without a readable number a player who appears in two shots
    is counted twice, which is honest: there is nothing in the picture that says they are
    the same person. Returns {track key: group id}.
    """
    keys = list(tracks)
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def shots_of(root):
        return {tracks[k].get("shot") for k in keys if find(k) == root}

    pairs = []
    for i, key in enumerate(keys):
        a = tracks[key]
        for other in keys[:i]:
            b = tracks[other]
            if a.get("shot") == b.get("shot"):
                continue
            number = a.get("number")
            if not number or number != b.get("number"):
                continue                               # no number, no evidence of identity
            if a.get("team") and b.get("team") and a["team"] != b["team"]:
                continue
            score = _similar(a.get("signature"), b.get("signature"))
            if score >= min_similarity:
                pairs.append((score, key, other))

    for _, key, other in sorted(pairs, reverse=True):
        root_a, root_b = find(key), find(other)
        if root_a == root_b or shots_of(root_a) & shots_of(root_b):
            continue                                   # one player cannot be in a shot twice
        parent[root_b] = root_a

    roots = {}
    return {k: roots.setdefault(find(k), len(roots)) for k in keys}
