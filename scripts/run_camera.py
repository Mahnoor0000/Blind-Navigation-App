import time
import threading
import queue
import subprocess

import cv2
from ultralytics import YOLO

from fusion_engine import FusionEngine
from caption import Captioner

# ---------- settings ----------
model_path = "models/best.pt"
camera_index = 0          # 0 = laptop webcam, try 1 for an external camera
image_size = 480          # lower = faster video (416 is faster still), 640 = best accuracy
min_conf = 0.45           # ignore detections the model is not sure about
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
voice = subprocess.Popen(
    ["powershell", "-NoProfile", "-NonInteractive", "-Command", speech_script],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
)

# speech runs in its own thread so the video does not freeze while talking
speech_queue = queue.Queue(maxsize=1)


def speaker_loop():
    while True:
        text = speech_queue.get()
        try:
            voice.stdin.write(text.replace("\n", " ") + "\n")
            voice.stdin.flush()
            voice.stdout.readline()    # waits until the sentence is finished
        except Exception as error:
            print("voice error:", error)


def say(text):
    # returns True if the text was accepted, False if the voice is still busy
    try:
        speech_queue.put_nowait(text)
        return True
    except queue.Full:
        return False


# ---------- captions ----------
# BLIP takes a few seconds, so it runs in a background thread
caption_state = {"busy": False, "text": "", "time": 0.0}


def caption_worker(frame):
    try:
        text = captioner.caption(frame)
    except Exception as error:
        print("caption error:", error)
        text = ""
    caption_state["text"] = text
    caption_state["time"] = time.time()
    caption_state["busy"] = False


# ---------- start ----------
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("could not open the camera, try camera_index = 1")
    raise SystemExit
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("loading YOLO...")
model = YOLO(model_path)
print("loading BLIP (the first time can take a while)...")
captioner = Captioner()

engine = FusionEngine(caption_interval=6.0)
threading.Thread(target=speaker_loop, daemon=True).start()

muted = False
shown_caption = ""
fps = 0.0
last_time = time.time()
print("running. press q to quit, m to mute the voice, c to turn captions on / off")

while True:
    ok, frame = cap.read()
    if not ok:
        print("lost the camera")
        break

    results = model(frame, imgsz=image_size, conf=min_conf, verbose=False)[0]
    height, width = frame.shape[:2]

    # turn the yolo boxes into simple dictionaries for the fusion engine
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "label": model.names[int(box.cls[0])],
            "area": ((x2 - x1) * (y2 - y1)) / (width * height),
            "x_center": (x1 + x2) / 2 / width,
        })

    # 1. urgent things first (hazards, new objects)
    phrases = engine.process(detections)
    if len(phrases) > 0 and not muted:
        if say(". ".join(phrases)):
            engine.confirm_spoken()
            print("saying:", ". ".join(phrases))

    # 2. ask BLIP for a description when it is due (in the background)
    if use_captions and not caption_state["busy"] and engine.caption_due():
        engine.caption_started()
        caption_state["busy"] = True
        threading.Thread(target=caption_worker, args=(frame.copy(),), daemon=True).start()

    # 3. speak a finished caption, but only if nothing urgent was said this frame
    if caption_state["text"] != "":
        text = caption_state["text"]
        age = time.time() - caption_state["time"]
        caption_state["text"] = ""
        if age < 6:
            text = engine.clean_caption(text)
            if text != "":
                shown_caption = text
                print("caption:", text)
                if len(phrases) == 0 and not muted:
                    say(text)

    # frames per second, so you can see how slow it really is
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 0.001))
    last_time = now

    # draw the video with the boxes, the fps and the last caption
    picture = results.plot()
    cv2.putText(picture, "FPS: " + str(int(fps)), (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.rectangle(picture, (0, height - 36), (width, height), (0, 0, 0), -1)
    cv2.putText(picture, shown_caption, (10, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Smart Blind Navigation", picture)

    pressed = cv2.waitKey(1) & 0xFF
    if pressed == ord("q"):
        break
    if pressed == ord("m"):
        muted = not muted
        print("muted" if muted else "voice on")
    if pressed == ord("c"):
        use_captions = not use_captions
        print("captions on" if use_captions else "captions off")

cap.release()
cv2.destroyAllWindows()
voice.terminate()