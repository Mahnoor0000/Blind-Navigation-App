import time

# things that can hurt the user, spoken first and repeated more often
HAZARD_CLASSES = {"person", "vehicle", "cycle", "stairs", "obstacle", "pole"}

# classes that are never spoken (tree is not reliable yet)
IGNORE_CLASSES = {"tree"}


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


class FusionEngine:
    def __init__(self, caption_interval=4.0, min_caption_gap=2.0,
                 hazard_cooldown=4.0, normal_cooldown=8.0, max_phrases=2):
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

    def process(self, detections, now=None):
        # detections is a list of dicts: {"label": "person", "area": 0.1, "x_center": 0.5}
        # returns the short phrases that should be spoken right now
        if now is None:
            now = time.time()

        usable = []
        for d in detections:
            if d["label"] not in IGNORE_CLASSES:
                usable.append(d)

        # did something new appear since the last frame?
        labels = set()
        for d in usable:
            labels.add(d["label"])
        self.new_objects = len(labels - self.last_seen_labels) > 0
        self.last_seen_labels = labels

        # hazards first, then the biggest (closest) ones
        usable.sort(key=lambda d: (d["label"] in HAZARD_CLASSES, d["area"]), reverse=True)

        phrases = []
        keys = []
        for d in usable:
            if len(phrases) >= self.max_phrases:
                break

            position = where(d["x_center"])
            key = d["label"] + "_" + position
            if key in keys:
                continue

            if d["label"] in HAZARD_CLASSES:
                cooldown = self.hazard_cooldown
            else:
                cooldown = self.normal_cooldown
            if key in self.last_spoken and now - self.last_spoken[key] < cooldown:
                continue

            phrase = d["label"].replace("_", " ") + " " + position
            distance = how_close(d["area"])
            if distance != "":
                phrase = phrase + ", " + distance
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

    def clean_caption(self, text):
        # returns "" if the caption should not be spoken
        text = text.strip()
        if text == "" or text == self.last_caption:
            return ""
        self.last_caption = text
        words = text.split()
        if len(words) > 14:
            text = " ".join(words[:14])
        return text