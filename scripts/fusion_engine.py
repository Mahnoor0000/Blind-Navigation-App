import re
import time
from collections import deque, Counter


# things that can hurt the user, spoken first and repeated more often
HAZARD_CLASSES = {"person", "vehicle", "stairs", "obstacle", "pole", "pothole",
                  "traffic_cone", "bench", "fire_hydrant", "dog", "animal"}

# classes that are never spoken (none at the moment; add a class here to silence it)
IGNORE_CLASSES = set()

# minimum confidence per class. Low for dangerous classes the model is unsure about
# (better a false alarm than a missed staircase), higher for everything else.
# run4 test: descending stairs found in 70% of images at 0.25 but only 40% at 0.5.
CLASS_CONF = {
    "stairs": 0.25,
    "pothole": 0.25,
    "person": 0.30,      # person recall is only ~0.60 (YOLO26n test), so accept less certain people
    "vehicle": 0.35,
    "obstacle": 0.35,
}
DEFAULT_CONF = 0.55       # raised from 0.45: live video gave false bench / parking meter
MIN_CONF = min(list(CLASS_CONF.values()) + [DEFAULT_CONF])   # run YOLO at this confidence

CONFUSED_WITH_STAIRS = {"traffic_light", "fire_hydrant"}

# spoken names that differ from the class name (light colours are no longer separate classes,
# so the model can never say "green light" when it is red)
SPEAK_AS = {}
# never say "up" or "down" for stairs: the model cannot tell. Add a warning word instead.
CAREFUL_CLASSES = {"stairs", "pothole"}

# words in a BLIP caption that must be backed up by YOLO, or the caption is dropped
# (BLIP invented a stop sign in the tests). word in caption -> YOLO classes that confirm it
CAPTION_CHECKS = {
    r"\bstop signs?\b": {"stop_sign"},
    r"\btraffic lights?\b": {"traffic_light"},
    r"\b(stairs?|staircases?|steps)\b": {"stairs"},
    r"\b(cars?|trucks?|bus|buses|motorbikes?|motorcycles?|bicycles?|bikes?)\b": {"vehicle"},
    r"\b(cows?|horses?|sheep|goats?|donkeys?|buffalo(es)?)\b": {"animal"},
    r"\b(dogs?|puppy|puppies)\b": {"dog"},
}
# BLIP guessed the wrong direction for stairs ("leading up" on stairs going down): cut it off
STAIR_DIRECTION = re.compile(r"\s*\b(leading|going|that lead|which lead)\s+(up|down)\b.*$")


def where(x_center):
    # x_center is 0 (left edge) to 1 (right edge) of the picture
    if x_center < 0.33:
        return "on your left"
    if x_center > 0.66:
        return "on your right"
    return "ahead"


def how_close(area):
    # area is how much of the picture the box covers, only a rough guess of distance
    if area > 0.25:
        return "very close"
    if area > 0.08:
        return "close"
    return ""


def keep_detections(detections):
    # applies the per-class thresholds and the stairs confusion rule
    kept = [d for d in detections
            if d["label"] not in IGNORE_CLASSES
            and d.get("conf", 1.0) >= CLASS_CONF.get(d["label"], DEFAULT_CONF)]
    if any(d["label"] == "stairs" for d in kept):
        kept = [d for d in kept if d["label"] not in CONFUSED_WITH_STAIRS]
    return kept


class FusionEngine:
    def __init__(self, caption_interval=4.0, min_caption_gap=2.0,
                 hazard_cooldown=4.0, normal_cooldown=8.0, max_phrases=2,
                 history=5, persist=3):
        self.caption_interval = caption_interval      # ask for a caption at least this often
        self.min_caption_gap = min_caption_gap        # but never more often than this
        self.hazard_cooldown = hazard_cooldown        # seconds before a hazard is repeated
        self.normal_cooldown = normal_cooldown        # seconds before anything else is repeated
        self.max_phrases = max_phrases                # most things said in one go

        self.last_caption_time = 0.0
        self.last_caption = ""
        self.last_seen_labels = set()
        self.new_objects = False
        self.last_spoken = {}
        self.pending_keys = []
        # an object is only announced if it was seen in `persist` of the last `history` frames.
        # a false detection usually flickers in for a single frame; a real object stays.
        self.history = deque(maxlen=history)
        self.persist = persist

    def process(self, detections, now=None):
        # detections: list of dicts {"label": "person", "conf": 0.6, "area": 0.1, "x_center": 0.5}
        # returns the short phrases that should be spoken right now
        if now is None:
            now = time.time()

        usable = keep_detections(detections)

        # temporal smoothing: drop labels that were not seen in enough recent frames
        self.history.append({d["label"] for d in usable})
        counts = Counter(label for frame_labels in self.history for label in frame_labels)
        usable = [d for d in usable if counts[d["label"]] >= self.persist]

        # did something new appear since the last frame?
        labels = {d["label"] for d in usable}
        self.new_objects = len(labels - self.last_seen_labels) > 0
        self.last_seen_labels = labels

        # hazards first, then the biggest (closest) ones
        usable.sort(key=lambda d: (d["label"] in HAZARD_CLASSES, d["area"]), reverse=True)

        phrases = []
        keys = []
        for d in usable:
            if len(phrases) >= self.max_phrases:
                break

            name = SPEAK_AS.get(d["label"], d["label"].replace("_", " "))
            position = where(d["x_center"])
            key = name + "_" + position
            if key in keys:
                continue

            if d["label"] in HAZARD_CLASSES:
                cooldown = self.hazard_cooldown
            else:
                cooldown = self.normal_cooldown
            if key in self.last_spoken and now - self.last_spoken[key] < cooldown:
                continue

            phrase = name + " " + position
            distance = how_close(d["area"])
            if distance != "":
                phrase = phrase + ", " + distance
            if d["label"] in CAREFUL_CLASSES:
                phrase = phrase + ", careful"
            phrases.append(phrase)
            keys.append(key)

        self.pending_keys = keys
        return phrases

    def confirm_spoken(self, now=None):
        # call this after the phrases from process() were really handed to the voice
        if now is None:
            now = time.time()
        for key in self.pending_keys:
            self.last_spoken[key] = now
        self.pending_keys = []

    def caption_due(self, now=None):
        if now is None:
            now = time.time()
        elapsed = now - self.last_caption_time
        if elapsed >= self.caption_interval:
            return True
        if self.new_objects and elapsed >= self.min_caption_gap:
            return True
        return False

    def caption_started(self, now=None):
        if now is None:
            now = time.time()
        self.last_caption_time = now

    def clean_caption(self, text, yolo_labels=None):
        # returns "" if the caption should not be spoken
        # yolo_labels: classes YOLO detected in the same frame BLIP described
        text = text.strip()
        if text == "" or text == self.last_caption:
            return ""

        if yolo_labels is not None:
            seen = set(yolo_labels)
            low = text.lower()
            for pattern, confirming in CAPTION_CHECKS.items():
                if re.search(pattern, low) and not (seen & confirming):
                    return ""        # BLIP names a hazard YOLO did not see: likely invented

        text = STAIR_DIRECTION.sub("", text)

        self.last_caption = text
        words = text.split()
        if len(words) > 14:
            text = " ".join(words[:14])
        return text