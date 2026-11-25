import argparse
import math
import threading
import time

import cv2
import numpy as np

from distance_fusion import DistanceEstimator


class DepthWorker(threading.Thread):
    def __init__(self, model):
        super().__init__(daemon=True)
        self.model, self.lock = model, threading.Lock()
        self.frame, self.depth, self.hfov, self.stamp, self.secs = None, None, None, 0.0, 0.0
        self.hfovs, self.running = [], True

    def submit(self, rgb):
        with self.lock:
            self.frame = rgb

    def latest(self):
        with self.lock:
            return self.depth, time.time() - self.stamp if self.stamp else 1e9

    def run(self):
        while self.running:
            with self.lock:
                rgb, self.frame = self.frame, None
            if rgb is None:
                time.sleep(0.01)
                continue
            t0 = time.perf_counter()
            depth, hfov = self.model(rgb)
            with self.lock:
                self.depth, self.stamp, self.secs = depth, time.time(), time.perf_counter() - t0
                if hfov:
                    self.hfovs.append(hfov)


def color_for(d):
    if d is None or not np.isfinite(d):
        return (200, 200, 200)
    return (0, 0, 255) if d < 1.5 else (0, 165, 255) if d < 3 else (0, 200, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--depth", default="moge", choices=["none", "moge", "da_indoor", "da_outdoor"])
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--hfov", type=float, default=65.0)
    ap.add_argument("--auto-fov", action="store_true")
    ap.add_argument("--cam-height", type=float, default=0.75)
    ap.add_argument("--depth-max-age", type=float, default=3.0)
    ap.add_argument("--da-size", type=int, default=308)
    ap.add_argument("--moge-tokens", type=int, default=1200)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO(args.yolo)

    worker = None
    if args.depth != "none":
        from compare_depth import DA_IDS, DepthAnythingMetric, MoGe2
        print(f"loading {args.depth} on {device} ...")
        model = MoGe2(device, args.moge_tokens) if args.depth == "moge" else \
            DepthAnythingMetric(DA_IDS[args.depth], device, args.da_size)
        worker = DepthWorker(model)
        worker.start()

    cap = cv2.VideoCapture(args.cam)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("cannot open webcam")
    H, W = frame.shape[:2]
    est = DistanceEstimator(W, H, hfov_deg=args.hfov, cam_height_m=args.cam_height)
    fov_set, last_print, t_prev = False, 0.0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth, age = (None, 1e9)
        if worker:
            worker.submit(rgb)
            depth, age = worker.latest()
            if age > args.depth_max_age:
                depth = None
            if args.auto_fov and not fov_set and len(worker.hfovs) >= 3:
                fov = float(np.median(worker.hfovs))
                est.f = (W / 2) / math.tan(math.radians(fov) / 2)
                fov_set = True
                print(f"camera FOV set from MoGe: {fov:.1f} deg")

        res = yolo.track(frame, persist=True, conf=args.conf, verbose=False)[0]
        outs = []
        ids = res.boxes.id.cpu().tolist() if res.boxes.id is not None else [None] * len(res.boxes)
        for b, c, tid in zip(res.boxes.xyxy.cpu().tolist(), res.boxes.cls.cpu().tolist(), ids):
            cls = res.names[int(c)]
            o = est.estimate(cls, b, depth_map=depth, track_id=tid)
            o["box"] = b
            outs.append(o)
            x1, y1, x2, y2 = map(int, b)
            d = o["distance_m"]
            col = color_for(d)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            label = f"{cls} {d:.1f}m {o['direction']}" if d else f"{cls} ? {o['direction']}"
            cv2.putText(frame, label, (x1 + 3, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cues = "  ".join(f"{k[0]}:{v:.1f}" for k, v in o["cues"].items() if v)
            cv2.putText(frame, cues, (x1 + 3, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        now = time.time()
        fps, t_prev = 1 / max(now - t_prev, 1e-6), now
        info = f"FPS {fps:.1f} | depth: {args.depth}"
        if worker:
            info += f" ({worker.secs:.1f}s/run, age {min(age, 99):.1f}s{'' if depth is not None else ' - not used'})"
        cv2.putText(frame, info, (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        near = [o for o in outs if o["distance_m"]]
        if near and now - last_print > 2:
            n = min(near, key=lambda o: o["distance_m"])
            print("SPEAK:", n["spoken"])
            last_print = now

        cv2.imshow("distance test (q quit, s snapshot)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            name = f"snap_{int(now)}.jpg"
            cv2.imwrite(name, frame)
            print(f"saved {name}")
            for o in outs:
                print(f"  {o['class']:<14} fused {o['distance_m']}  cues {o['cues']}  {o['direction']}")

    if worker:
        worker.running = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()