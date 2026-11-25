import math
import numpy as np

# Rough real-world heights (m). Only classes with predictable height; others skip the pinhole cue.
KNOWN_HEIGHT_M = {
    "person": 1.65, "door": 2.05, "vehicle": 1.5, "traffic_cone": 0.7, "fire_hydrant": 0.75,
    "stop_sign": 2.1, "traffic_light": 3.0, "chair": 0.9, "table": 0.75, "bin": 0.9,
    "fire_extinguisher": 0.55, "elevator": 2.1, "dog": 0.55, "pole": None,  # pole often cut off
}
# Classes that touch the ground (ground-contact cue valid). Signs/lights hang, so excluded.
GROUND_CLASSES = {
    "person", "vehicle", "chair", "table", "bin", "door", "obstacle", "traffic_cone", "fire_hydrant",
    "bench", "dog", "animal", "pole", "stairs", "pothole", "crosswalk", "shelf", "elevator",
}


class DistanceEstimator:
    def __init__(self, img_w, img_h, hfov_deg=66.0, cam_height_m=1.3, horizon_row=None,
                 ema_alpha=0.4, max_dist_m=15.0):
        self.w, self.h = img_w, img_h
        # focal length in pixels from horizontal field of view (phone main cams ~ 65-78 deg)
        self.f = (img_w / 2) / math.tan(math.radians(hfov_deg) / 2)
        self.cam_h = cam_height_m
        # horizon row: image centre if phone held upright & level; override from accelerometer pitch
        self.horizon = img_h / 2 if horizon_row is None else horizon_row
        self.alpha, self.max_d = ema_alpha, max_dist_m
        self._smooth = {}  # track_id -> smoothed distance

    def set_pitch(self, pitch_deg):
        """pitch>0 = camera tilted down. Moves horizon row up in the image."""
        self.horizon = self.h / 2 - self.f * math.tan(math.radians(pitch_deg))

    # ---- cues -------------------------------------------------------------
    def ground_contact(self, cls, box):
        x1, y1, x2, y2 = box
        if cls not in GROUND_CLASSES or y2 >= self.h - 2:  # bottom cut off -> contact point unseen
            return None
        dy = y2 - self.horizon
        if dy <= 5:                                         # at/above horizon -> unreliable
            return None
        return self.f * self.cam_h / dy

    def pinhole(self, cls, box):
        H = KNOWN_HEIGHT_M.get(cls)
        x1, y1, x2, y2 = box
        if not H or y1 <= 2 or y2 >= self.h - 2:           # truncated box -> height wrong
            return None
        bh = y2 - y1
        return self.f * H / bh if bh > 8 else None

    @staticmethod
    def depth_cue(box, depth_map):
        if depth_map is None:
            return None
        x1, y1, x2, y2 = map(int, box)
        bw, bh = x2 - x1, y2 - y1
        # lower-middle patch: avoids background leaking through box edges/top
        patch = depth_map[y1 + int(0.5 * bh): y2, x1 + int(0.25 * bw): x2 - int(0.25 * bw)]
        patch = patch[np.isfinite(patch) & (patch > 0)]
        return float(np.median(patch)) if patch.size > 10 else None

    # ---- fusion -----------------------------------------------------------
    def direction(self, box):
        cx = (box[0] + box[2]) / 2 / self.w
        if cx < 0.35:
            return "on your left"
        if cx > 0.65:
            return "on your right"
        return "ahead"

    def estimate(self, cls, box, depth_map=None, track_id=None):
        cues = {
            "ground": (self.ground_contact(cls, box), 1.0),
            "pinhole": (self.pinhole(cls, box), 0.8),
            "depth": (self.depth_cue(box, depth_map), 1.2),
        }
        vals = [(v, w) for v, w in cues.values() if v is not None and 0.2 < v < 60]
        if not vals:
            d = None
        else:
            vals.sort()
            half, acc = sum(w for _, w in vals) / 2, 0.0
            for v, w in vals:                               # weighted median: robust to one bad cue
                acc += w
                if acc >= half:
                    d = v
                    break
            if track_id is not None:                        # temporal smoothing for video
                prev = self._smooth.get(track_id)
                d = d if prev is None else self.alpha * d + (1 - self.alpha) * prev
                self._smooth[track_id] = d
        direction = self.direction(box)
        return {
            "class": cls, "distance_m": d, "direction": direction,
            "cues": {k: (round(v, 2) if v else None) for k, (v, _) in cues.items()},
            "spoken": self.speak(cls, d, direction),
        }

    def speak(self, cls, d, direction):
        name = cls.replace("_", " ")
        if d is None:
            return f"{name} {direction}"
        if d < 1.0:
            return f"{name} very close, {direction}"
        if d > self.max_d:
            return f"{name} far {direction}"
        rounded = round(d * 2) / 2 if d < 5 else round(d)  # 0.5 m steps near, 1 m steps far
        return f"{name} {direction}, about {rounded:g} metres"


if __name__ == "__main__":
    # sanity check: 640x480 frame, person 3 m away, phone at 1.3 m, level
    est = DistanceEstimator(640, 480, hfov_deg=66, cam_height_m=1.3)
    f, Z = est.f, 3.0
    y2 = est.horizon + f * 1.3 / Z
    y1 = y2 - f * 1.65 / Z
    out = est.estimate("person", (300, y1, 360, y2))
    print(out)