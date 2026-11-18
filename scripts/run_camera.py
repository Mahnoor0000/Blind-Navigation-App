import os
import time
import threading
import queue
import subprocess

import cv2
from ultralytics import YOLO

from fusion_engine import FusionEngine, keep_detections, MIN_CONF, LANGUAGE
from caption import Captioner
from spatial import Spatial, Tracker

# ---------- settings ----------
model_path = "models/yolo_26n_clean.pt"   # the 21-class YOLO26n (mAP50 0.746)
camera_index = 0          # 0 = laptop webcam, a phone stream like "http://192.168.1.23:4747/video",
                          # or a VIDEO FILE like "test_videos/walk1.mp4" (played frame by frame)
save_video = True         # for video files: also write the annotated result to test_results/

# distance / direction (spatial.py). Measure these for your setup, they matter:
HFOV_DEG = 70             # camera field of view along the long side (phone main camera ~65-80, webcam ~60-70)
CAMERA_HEIGHT = 1.3       # metres from the floor to the camera (phone at chest height ~1.3)
CAMERA_PITCH_DEG = 10     # starting guess: how far the camera points DOWN; learned from people / doors / cars
AUTO_PITCH = True
image_size = 480          # lower = faster video (416 is faster still), 640 = best accuracy
min_conf = MIN_CONF       # lowest per-class threshold; fusion_engine applies the real per-class ones
use_captions = True       # BLIP scene descriptions, press c to turn them on / off


# ---------- voice ----------
# uses the voice that is built into Windows. one PowerShell process stays open,
# we send it a line of text and it speaks it, then replies "done"
speech_script = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$s.Rate = 1; "
    "while (($line = [Console]::In.ReadLine()) -ne $null) { "
    "$s.Speak($line); [Console]::Out.WriteLine('done'); [Console]::Out.Flush() }"
)


def start_voice():
    return subprocess.Popen(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", speech_script],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
    )


voice = start_voice()
running = True

# Urdu warnings are pre-made clips (python scripts/urdu_voice.py); captions stay English (BLIP is English)
urdu = None
if LANGUAGE == "ur":
    from urdu_voice import ClipPlayer
    urdu = ClipPlayer()
    missing = len(ClipPlayer.missing())
    if missing:
        print(f"WARNING: {missing} Urdu clips missing - run: python scripts/urdu_voice.py")

# speech runs in its own thread so the video does not freeze while talking
speech_queue = queue.Queue(maxsize=1)
voice_busy = threading.Event()     # set while a sentence is being spoken


def speaker_loop():
    global voice
    while running:
        text = speech_queue.get()
        voice_busy.set()
        if isinstance(text, list):          # Urdu warnings: play one clip after another
            for phrase in text:
                urdu.play(phrase)
            voice_busy.clear()
            continue
        try:
            voice.stdin.write(text.replace("\n", " ") + "\n")
            voice.stdin.flush()
            voice.stdout.readline()    # waits until the sentence is finished
        except Exception as error:
            if running:
                # the PowerShell voice died: start a new one and carry on
                print("voice error, restarting the voice:", error)
                try:
                    voice.kill()
                except Exception:
                    pass
                voice = start_voice()
        voice_busy.clear()


def say(text):
    # returns True if the text was accepted, False if the voice is still busy
    if voice_busy.is_set():
        return False
    try:
        speech_queue.put_nowait(text)
        return True
    except queue.Full:
        return False


# ---------- captions ----------
# BLIP takes a few seconds, so it runs in a background thread
caption_state = {"busy": False, "text": "", "time": 0.0, "labels": set()}


def caption_worker(frame, labels):
    try:
        text = captioner.caption(frame)
    except Exception as error:
        print("caption error:", error)
        text = ""
    caption_state["labels"] = labels   # what YOLO saw in the same frame, to check the caption
    caption_state["text"] = text
    caption_state["time"] = time.time()
    caption_state["busy"] = False


# ---------- camera ----------
# reads frames in its own thread and always keeps only the newest one.
# works for a webcam number (0, 1) and for a phone stream URL (DroidCam, IP Webcam).
# if the stream drops, it reconnects instead of stopping the program.
class Camera:
    def __init__(self, source):
        self.source = source
        self.frame = None
        self.lock = threading.Lock()
        self.cap = None
        threading.Thread(target=self._reader, daemon=True).start()

    def _open(self):
        cap = cv2.VideoCapture(self.source)
        if isinstance(self.source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _reader(self):
        while running:
            if self.cap is None or not self.cap.isOpened():
                self.cap = self._open()
                if not self.cap.isOpened():
                    print("camera not reachable, retrying in 2 s...")
                    time.sleep(2)
                    continue
                print("camera connected")
            try:
                ok, frame = self.cap.read()
            except cv2.error as error:
                # OpenCV sometimes throws on a broken network frame; treat it as a dropped stream
                print("camera read error:", error)
                ok = False
            if not ok:
                print("camera stream dropped, reconnecting...")
                self.cap.release()
                self.cap = None
                time.sleep(1)
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        if self.cap is not None:
            self.cap.release()


class VideoFile:
    # plays a recorded video frame by frame (nothing is skipped), with the video's own clock,
    # so time-to-contact is computed in real video seconds even if processing is slower
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.index = 0
        self.done = False

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            self.done = True
            return None
        self.index += 1
        return frame

    def time(self):
        return self.index / self.fps

    def release(self):
        self.cap.release()


is_file = isinstance(camera_index, str) and os.path.isfile(camera_index)

print("loading YOLO...")
model = YOLO(model_path)
print("loading BLIP (the first time can take a while)...")
captioner = Captioner()

engine = FusionEngine(caption_interval=6.0)
spatial = Spatial(hfov_deg=HFOV_DEG, cam_height=CAMERA_HEIGHT, pitch_deg=CAMERA_PITCH_DEG, auto_pitch=AUTO_PITCH)
tracker = Tracker()
writer = None
threading.Thread(target=speaker_loop, daemon=True).start()

muted = False
shown_caption = ""
pending_caption = ""
pending_caption_time = 0.0
fps = 0.0
last_time = time.time()
# open the camera only now, after the slow model loading, so a phone stream does not time out
cam = VideoFile(camera_index) if is_file else Camera(camera_index)
print("running. press q to quit, m to mute, c captions on / off, s to save a frame")

while True:
    frame = cam.read()
    if is_file and cam.done:
        print("end of video")
        break
    if frame is None:
        # no picture yet (still connecting); keep the window responsive
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break
        continue

    results = model(frame, imgsz=image_size, conf=min_conf, verbose=False)[0]
    height, width = frame.shape[:2]

    t_now = cam.time() if is_file else time.time()

    # turn the yolo boxes into dictionaries for the fusion engine, with distance and direction
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = model.names[int(box.cls[0])]
        d = {
            "label": label,
            "conf": float(box.conf[0]),
            "area": ((x2 - x1) * (y2 - y1)) / (width * height),
            "x_center": (x1 + x2) / 2 / width,
            "box": (x1, y1, x2, y2),
        }
        d.update(spatial.measure(x1, y1, x2, y2, label, width, height))
        detections.append(d)
    # learn the camera tilt from confident known-size objects, then smooth / track over frames
    spatial.calibrate_pitch([d["pitch_obs"] for d in detections if d["conf"] >= 0.5])
    tracker.update(detections, t_now)

    # 1. urgent things first (in your path and close, approaching, hazards)
    phrases = engine.process(detections, now=t_now if is_file else None)
    if len(phrases) > 0 and not muted:
        if say(phrases if urdu is not None else ". ".join(phrases)):
            engine.confirm_spoken()
            print("saying:", " | ".join(phrases))

    # 2. ask BLIP for a description when it is due (in the background)
    if use_captions and not caption_state["busy"] and engine.caption_due():
        engine.caption_started()
        caption_state["busy"] = True
        frame_labels = {d["label"] for d in keep_detections(detections)}
        threading.Thread(target=caption_worker, args=(frame.copy(), frame_labels), daemon=True).start()

    # 3. speak a finished caption, but only if nothing urgent was said this frame
    #    (captions are checked against YOLO: a caption naming a hazard YOLO did not see is dropped)
    if caption_state["text"] != "":
        text = engine.clean_caption(caption_state["text"], caption_state["labels"])
        caption_state["text"] = ""
        if text != "":
            shown_caption = text
            print("caption:", text)
            pending_caption = text              # keep it until the voice is free
            pending_caption_time = caption_state["time"]

    # a caption waits while hazards are being spoken, and is thrown away if it gets old
    if pending_caption != "":
        if time.time() - pending_caption_time > 8:
            pending_caption = ""                # the scene has probably changed
        elif len(phrases) == 0 and not muted and say(pending_caption):
            print("saying caption:", pending_caption)
            pending_caption = ""

    # frames per second, so you can see how slow it really is
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 0.001))
    last_time = now

    # draw the video: boxes with distance (red = in your path), fps, tilt and the last caption
    picture = frame.copy()
    for d in keep_detections(detections):
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        colour = (0, 0, 255) if d["in_path"] else (0, 200, 0)
        dist = d.get("distance_smooth")
        text = d["label"]
        if dist:
            text += f" {dist:.1f}m" + ("?" if d["rough"] else "")
        if d.get("ttc") is not None and d["ttc"] < 4:
            text += f" {d['ttc']:.1f}s"
        cv2.rectangle(picture, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(picture, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
    # the walking corridor at 3 m, to see what "in your path" means
    f = spatial.focal(width, height)
    half = int(f * spatial.path_half_width / 3)
    cv2.line(picture, (width // 2 - half, height), (width // 2 - half, height // 2), (255, 255, 0), 1)
    cv2.line(picture, (width // 2 + half, height), (width // 2 + half, height // 2), (255, 255, 0), 1)
    cv2.putText(picture, f"FPS: {int(fps)}  tilt: {spatial.pitch_deg:.0f} deg", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.rectangle(picture, (0, height - 36), (width, height), (0, 0, 0), -1)
    cv2.putText(picture, shown_caption, (10, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Smart Blind Navigation", picture)
    if is_file and save_video:
        if writer is None:
            os.makedirs("test_results", exist_ok=True)
            out_path = os.path.join("test_results", "annotated_" + os.path.basename(camera_index))
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), cam.fps, (width, height))
            print("saving annotated video to", out_path)
        writer.write(picture)

    pressed = cv2.waitKey(1) & 0xFF
    if pressed == ord("q"):
        break
    if pressed == ord("m"):
        muted = not muted
        print("muted" if muted else "voice on")
    if pressed == ord("c"):
        use_captions = not use_captions
        print("captions on" if use_captions else "captions off")
    if pressed == ord("s"):
        # save this frame and the last caption, to collect examples of bad captions
        os.makedirs("live_samples", exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join("live_samples", stamp + ".jpg"), frame)
        with open(os.path.join("live_samples", "captions.csv"), "a", encoding="utf-8") as f:
            f.write(stamp + ".jpg," + '"' + shown_caption.replace('"', "'") + '"\n')
        print("saved live_samples/" + stamp + ".jpg with caption:", shown_caption)

running = False
if writer is not None:
    writer.release()
cam.release()
cv2.destroyAllWindows()
voice.terminate()