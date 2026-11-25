import math
import re
import time
from collections import deque, Counter

from spatial import spoken_distance

# Class list (YOLO26n, merged_dataset_v3, 21 classes):
#   person, vehicle, pole, stairs, door, chair, table, shelf, bin, fire_extinguisher, obstacle,
#   traffic_light, stop_sign, fire_hydrant, bench, dog, pothole, crosswalk, traffic_cone,
#   elevator, animal

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

# the old model confused these with stairs (green light 0.65, fire hydrant 0.58 on stair photos),
# so they are dropped whenever stairs are in the same frame
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


# ---------- spoken language ----------
# "en" = English, "ur" = Urdu. Urdu needs the pre-made voice clips (python scripts/urdu_voice.py)
LANGUAGE = "en"

# Urdu words for the 21 classes (check with native speakers; easy to change here)
UR_NAMES = {
    "person": "شخص", "vehicle": "گاڑی", "pole": "کھمبا", "stairs": "سیڑھیاں",
    "door": "دروازہ", "chair": "کرسی", "table": "میز", "shelf": "شیلف", "bin": "کوڑے دان",
    "fire_extinguisher": "آگ بجھانے کا آلہ", "obstacle": "رکاوٹ", "traffic_light": "ٹریفک سگنل",
    "stop_sign": "اسٹاپ کا نشان", "fire_hydrant": "فائر ہائیڈرنٹ", "bench": "بینچ", "dog": "کتا",
    "pothole": "گڑھا", "crosswalk": "زیبرا کراسنگ", "traffic_cone": "ٹریفک کون",
    "elevator": "لفٹ", "animal": "جانور",
}
POSITION_WORDS = {
    "en": {"left": "on your left", "right": "on your right", "ahead": "ahead"},
    "ur": {"left": "بائیں طرف", "right": "دائیں طرف", "ahead": "سامنے"},
}
DISTANCE_WORDS = {
    "en": {"very_close": "very close", "close": "close", "": ""},
    "ur": {"very_close": "بہت قریب", "close": "قریب", "": ""},
}
CAREFUL_WORD = {"en": "careful", "ur": "احتیاط کریں"}


def where(x_center):
    # x_center is 0 (left edge) to 1 (right edge) of the picture
    if x_center < 0.33:
        return "left"
    if x_center > 0.66:
        return "right"
    return "ahead"


def how_close(area):
    # area is how much of the picture the box covers, only a rough guess of distance
    if area > 0.25:
        return "very_close"
    if area > 0.08:
        return "close"
    return ""


def make_phrase(label, position, distance, lang=None):
    # e.g. en: "stairs ahead, very close, careful"
    #      ur: "سامنے سیڑھیاں، بہت قریب، احتیاط کریں"   (Urdu puts the position first)
    lang = lang or LANGUAGE
    if lang == "ur":
        name = UR_NAMES.get(label, label.replace("_", " "))
        parts = [POSITION_WORDS["ur"][position] + " " + name]
    else:
        name = SPEAK_AS.get(label, label.replace("_", " "))
        parts = [name + " " + POSITION_WORDS["en"][position]]
    if distance:
        parts.append(DISTANCE_WORDS[lang][distance])
    if label in CAREFUL_CLASSES:
        parts.append(CAREFUL_WORD[lang])
    return ("، " if lang == "ur" else ", ").join(parts)


SIDE_WORDS = {"ahead": "ahead", "slightly left": "slightly left", "left": "left",
              "slightly right": "slightly right", "right": "right"}


def spatial_phrase(d):
    # short on purpose (every word costs speaking time while walking), e.g.
    #   "person ahead, 2 metres, approaching"   "chair slightly left, 3 metres"   "stairs right, close, careful"
    # "ahead" means in your walking path (spatial.py)
    name = SPEAK_AS.get(d["label"], d["label"].replace("_", " "))
    parts = [name + " " + SIDE_WORDS[d["side"]]]
    dist = d.get("distance_smooth") or d.get("distance")
    if dist:
        if d.get("at_most") and not d.get("d_box") and not d.get("d_ground"):
            # the object goes below the picture: we only know it is closer than the bottom edge
            limit = math.ceil(d["at_most"])
            parts.append("very close" if limit <= 1 else f"within {limit} metres")
        else:
            parts.append(spoken_distance(dist, rough=d.get("rough", False)))
    ttc = d.get("ttc")
    if ttc is not None and ttc < 3 and d.get("in_path"):
        parts.append("approaching")
    if d["label"] in CAREFUL_CLASSES:
        parts.append("careful")
    return ", ".join(parts)


def is_urgent(d):
    # in the walking path and close, or getting close fast
    dist = d.get("distance_smooth") or d.get("distance")
    return bool(d.get("in_path")) and ((dist is not None and dist < 2) or
                                       (d.get("ttc") is not None and d["ttc"] < 2.5))


def all_phrases(lang):
    # every phrase the engine can say (21 classes x 3 positions x 3 distances), for pre-made voice clips
    out = []
    for label in UR_NAMES:
        for position in ("left", "ahead", "right"):
            for distance in ("", "close", "very_close"):
                out.append(make_phrase(label, position, distance, lang))
    return out


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
                 history=5, persist=3, lang=None, urgent_cooldown=2.0):
        self.urgent_cooldown = urgent_cooldown        # things in your path and close are repeated sooner
        self.lang = lang or LANGUAGE
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
        self.last_phrase = {}                         # what was said for each object, to avoid repeats
        self.pending_phrases = {}
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

        # with distances (spatial.py): urgent first, then in your path, hazards, nearest
        # without: hazards first, then the biggest (closest) ones
        def priority(d):
            if "side" in d:
                dist = d.get("distance_smooth") or d.get("distance") or 99
                return (is_urgent(d), bool(d.get("in_path")), d["label"] in HAZARD_CLASSES, -dist)
            return (False, False, d["label"] in HAZARD_CLASSES, d["area"])
        usable.sort(key=priority, reverse=True)

        phrases = []
        keys = []
        for d in usable:
            if len(phrases) >= self.max_phrases:
                break

            position = d["side"] if "side" in d else where(d["x_center"])
            key = d["label"] + "_" + position
            if key in keys:
                continue

            phrase = (spatial_phrase(d) if "side" in d and self.lang == "en"
                      else make_phrase(d["label"], where(d["x_center"]), how_close(d["area"]), self.lang))
            changed = self.last_phrase.get(key) != phrase
            if "side" in d and is_urgent(d) and changed:
                cooldown = self.urgent_cooldown       # quick repeat only if something changed
            elif d["label"] in HAZARD_CLASSES:
                cooldown = self.hazard_cooldown
            else:
                cooldown = self.normal_cooldown
            if key in self.last_spoken and now - self.last_spoken[key] < cooldown:
                continue

            phrases.append(phrase)
            keys.append(key)
            self.pending_phrases[key] = phrase

        self.pending_keys = keys
        return phrases

    def confirm_spoken(self, now=None):
        # call this after the phrases from process() were really handed to the voice
        if now is None:
            now = time.time()
        for key in self.pending_keys:
            self.last_spoken[key] = now
            if key in self.pending_phrases:
                self.last_phrase[key] = self.pending_phrases[key]
        self.pending_phrases = {}
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
        if len(words) > 25:                  # v2 captions describe the whole scene (up to ~25 words)
            text = " ".join(words[:25])
        return text