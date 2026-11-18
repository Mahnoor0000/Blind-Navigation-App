import math
from collections import deque

# real heights in metres (only classes with a fairly fixed size)
KNOWN_HEIGHT = {
    "person": 1.65, "door": 2.0, "vehicle": 1.5, "elevator": 2.1, "fire_hydrant": 0.75,
    "traffic_cone": 0.7, "bin": 0.9, "chair": 0.9, "bench": 0.8, "dog": 0.6,
}
# classes whose box bottom touches the floor (signs, lights and wall extinguishers do not)
ON_GROUND = {
    "person", "vehicle", "pole", "stairs", "door", "chair", "table", "shelf", "bin", "obstacle",
    "fire_hydrant", "bench", "dog", "pothole", "crosswalk", "traffic_cone", "elevator", "animal",
}
MIN_PERSON_ASPECT = 2.0     # box height / width below this = sitting or half-hidden person
CALIB_LABELS = {"person", "door", "vehicle"}   # sizes reliable enough to learn the camera tilt from


def median(values):
    v = sorted(values)
    n = len(v)
    return None if n == 0 else (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]))


def spoken_distance(m, rough=False):
    # short and rounded (speech must be quick); when the cues disagree, only a coarse range
    if rough:
        return "very close" if m < 1.5 else "close" if m < 3 else "a few metres" if m < 6 else "far"
    if m < 1:
        return "under 1 metre"
    r = round(m * 2) / 2 if m < 4 else round(m)
    return f"{r:g} metre" + ("" if r == 1 else "s")


class Spatial:
    def __init__(self, hfov_deg=70.0, cam_height=1.3, pitch_deg=10.0, auto_pitch=True,
                 path_half_width=0.45, max_disagree=2.0):
        self.hfov_deg = hfov_deg              # camera field of view along the LONG side of the image
        self.cam_height = cam_height          # metres above the floor (phone held at chest ~1.3 m)
        self.pitch_deg = pitch_deg            # how far the camera is tilted DOWN (degrees)
        self.auto_pitch = auto_pitch
        self.path_half_width = path_half_width
        self.max_disagree = max_disagree      # cues further apart than this factor -> "rough"

    def focal(self, W, H):
        return max(W, H) / (2 * math.tan(math.radians(self.hfov_deg / 2)))

    def measure(self, x1, y1, x2, y2, label, W, H, extra=()):
        f = self.focal(W, H)
        cx, cy = W / 2, H / 2
        w, h = x2 - x1, y2 - y1
        cut_top, cut_bottom = y1 <= 2, y2 >= H - 3
        out = {"label": label}

        # 1. known size
        d_box = None
        standing = not (label == "person" and h < MIN_PERSON_ASPECT * w)
        if label in KNOWN_HEIGHT and not cut_top and not cut_bottom and standing:
            d_box = f * KNOWN_HEIGHT[label] / max(h, 1)

        # 2. ground contact (bottom of the box = where it touches the floor)
        d_ground, at_most = None, None
        if label in ON_GROUND:
            angle = math.atan((y2 - cy) / f) + math.radians(self.pitch_deg)
            if angle > math.radians(1.5):            # at or above the horizon: no floor contact visible
                d = self.cam_height / math.tan(angle)
                if cut_bottom:
                    at_most = d                       # object continues below the picture: closer than this
                else:
                    d_ground = d

        # what tilt would make the ground cue agree with the box-height cue (for auto calibration)
        out["pitch_obs"] = None
        if d_box and label in CALIB_LABELS and not cut_bottom:
            out["pitch_obs"] = math.degrees(math.atan(self.cam_height / d_box) - math.atan((y2 - cy) / f))

        # 3. fuse
        cues = [d for d in (d_box, d_ground, *extra) if d and d > 0]
        distance = median(cues)
        rough = len(cues) >= 2 and max(cues) / min(cues) > self.max_disagree
        if distance is None and at_most:
            distance, rough = at_most, False
        out.update(d_box=d_box, d_ground=d_ground, at_most=at_most, distance=distance,
                   rough=rough, n_cues=len(cues))

        # 4. direction: angles of the box edges, then metres left / right of the walking line
        a_left = math.atan((x1 - cx) / f)
        a_right = math.atan((x2 - cx) / f)
        a_mid = math.degrees(math.atan(((x1 + x2) / 2 - cx) / f))
        if distance:
            left_m, right_m = distance * math.tan(a_left), distance * math.tan(a_right)
            in_path = left_m < self.path_half_width and right_m > -self.path_half_width
            offset = 0.0 if in_path else (left_m if left_m > 0 else right_m)
        else:
            in_path = x1 < cx + 0.1 * W and x2 > cx - 0.1 * W     # covers the centre of the picture
            offset = None
        if in_path:
            side = "ahead"
        elif a_mid < 0:
            side = "slightly left" if a_mid > -15 else "left"
        else:
            side = "slightly right" if a_mid < 15 else "right"
        hour = round(a_mid / 30)                       # 30 degrees per clock hour, 12 = straight ahead
        out.update(angle=a_mid, in_path=in_path, side=side, offset_m=offset,
                   clock=12 if hour == 0 else (hour if hour > 0 else 12 + hour))
        return out

    def calibrate_pitch(self, observations, alpha=0.15):
        # smooth the camera tilt towards what the known-size objects say
        obs = [p for p in observations if p is not None and -20 < p < 60]
        if self.auto_pitch and obs:
            self.pitch_deg = (1 - alpha) * self.pitch_deg + alpha * median(obs)
        return self.pitch_deg


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Tracker:
    # follows objects between frames (same label + overlapping boxes)
    def __init__(self, max_age=0.8, window=1.5, smooth=5):
        self.tracks = []          # each: {"id", "label", "box", "seen", "hist": deque[(t, h_px, dist)]}
        self.next_id = 0
        self.max_age, self.window, self.smooth = max_age, window, smooth

    def update(self, dets, t):
        # dets: list of dicts with "label", "box" (x1, y1, x2, y2) and "distance"; adds
        # "track", "distance_smooth" and "ttc" (seconds to contact, or None) to each
        free = list(self.tracks)
        for d in sorted(dets, key=lambda d: -(d["box"][3] - d["box"][1])):
            best, best_iou = None, 0.25
            for tr in free:
                if tr["label"] == d["label"]:
                    o = iou(tr["box"], d["box"])
                    if o > best_iou:
                        best, best_iou = tr, o
            if best is None:
                best = {"id": self.next_id, "label": d["label"], "hist": deque(maxlen=60)}
                self.next_id += 1
                self.tracks.append(best)
            else:
                free.remove(best)
            best["box"], best["seen"] = d["box"], t
            best["hist"].append((t, d["box"][3] - d["box"][1], d.get("distance")))

            recent = [x for x in best["hist"] if t - x[0] <= self.window]
            dists = [x[2] for x in list(best["hist"])[-self.smooth:] if x[2]]
            d["track"] = best["id"]
            d["distance_smooth"] = median(dists) if dists else d.get("distance")
            d["ttc"] = None
            if len(recent) >= 4 and recent[-1][0] - recent[0][0] >= 0.4:
                (t0, h0, _), (t1, h1, _) = recent[0], recent[-1]
                growth = (h1 - h0) / (t1 - t0)                 # pixels per second
                if h1 > 1.05 * h0 and growth > 0:              # clearly getting bigger = getting closer
                    d["ttc"] = h1 / growth                     # time-to-contact, seconds
        self.tracks = [tr for tr in self.tracks if t - tr["seen"] <= self.max_age]
        return dets