import os
import time

import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# fine-tuned on VizWiz; best on the 31 own photos with setting A (CIDEr 1.019)
model_name = "models/blip_qwen_drop"
PROMPT = None
GEN_OPTIONS = dict(num_beams=1, max_new_tokens=30, no_repeat_ngram_size=3)


class Captioner:
    def __init__(self, name=model_name, prompt=PROMPT, options=None):
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
        return text.rstrip(".")


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