import os
import time
import re
import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# BLIP-base: VizWiz fine-tune, then fine-tuned on Qwen2.5-VL v3 scene captions (no YOLO hints).
# Chosen over models/blip_vizwiz after comparing both on the 32 own test photos.
# To switch back, set MODEL_PATH = "models/blip_vizwiz".
MODEL_PATH = "models/blip_nav_v3"
PROMPT = None
# 3 beams + max_length 45 = the settings used in the 32-photo comparison
# (new captions are ~15 words, so a shorter limit cuts them off).
# repetition_penalty fixes repeats like "power lines, and power lines".
GEN_OPTIONS = dict(num_beams=3, max_length=45, no_repeat_ngram_size=3, repetition_penalty=1.3)


def clean_caption(text):
    # cut-off sign reading at the end: "... and a sign that says" / "... a sign that says blokk"
    text = re.sub(r",?\s*(and )?(a|the)?\s*sign that says\s*\w{0,6}$", "", text)
    # tokenizer spacing: "man ' s" -> "man's", "double - decker" -> "double-decker"
    text = text.replace(" ' s", "'s").replace(" - ", "-")
    # trailing comma / "and" left after cutting
    text = re.sub(r"(,|\band|\bwith)\s*$", "", text).strip()
    return text.rstrip(".")


class Captioner:
    def __init__(self, name=MODEL_PATH, prompt=PROMPT, options=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompt = prompt
        self.options = options or GEN_OPTIONS
        self.processor = BlipProcessor.from_pretrained(name)
        self.model = BlipForConditionalGeneration.from_pretrained(name).to(self.device)
        self.model.eval()

    def caption(self, frame_bgr):
        # frame_bgr is a normal opencv image (camera frame or cv2.imread)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        if self.prompt:
            inputs = self.processor(images=image, text=self.prompt, return_tensors="pt")
        else:
            inputs = self.processor(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, **self.options)
        text = self.processor.decode(out[0], skip_special_tokens=True).strip()
        # with a prompt, the prompt is part of the output: "a view of a room ..." -> "a room ..."
        if self.prompt and text.lower().startswith(self.prompt.lower()):
            text = text[len(self.prompt):].strip()
        return clean_caption(text)


# running this file directly captions every image in the caption test folder
if __name__ == "__main__":
    image_folder = "TEST IMAGES FOR BLIP"
    captioner = Captioner()

    for fname in sorted(os.listdir(image_folder)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image = cv2.imread(os.path.join(image_folder, fname))
        if image is None:
            continue
        start = time.time()
        text = captioner.caption(image)
        elapsed = time.time() - start
        print(fname, "-", text, "-", round(elapsed, 2), "seconds")