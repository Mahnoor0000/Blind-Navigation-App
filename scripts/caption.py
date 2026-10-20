import os
import time

import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

model_name = "Salesforce/blip-image-captioning-base"   # later: path to the fine-tuned "best" folder

# setting F from the sweep: fewest invented objects, good CIDEr, faster than 3-beam without prompt
PROMPT = "a view of"
GEN_OPTIONS = dict(num_beams=3, min_new_tokens=6, max_new_tokens=30,
                   repetition_penalty=1.3, no_repeat_ngram_size=3)


class Captioner:
    def __init__(self, name=model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(name)
        self.model = BlipForConditionalGeneration.from_pretrained(name).to(self.device)
        self.model.eval()

    def caption(self, frame_bgr):
        # frame_bgr is a normal opencv image (camera frame or cv2.imread)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        inputs = self.processor(images=image, text=PROMPT, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, **GEN_OPTIONS)
        text = self.processor.decode(out[0], skip_special_tokens=True).strip()
        # the prompt is part of the output: "a view of a lecture room ..." -> "a lecture room ..."
        if text.lower().startswith(PROMPT):
            text = text[len(PROMPT):].strip()
        return text


# running this file directly captions every image in test_images (your old test)
if __name__ == "__main__":
    image_folder = "TEST IMAGES FOR BLIP"
    captioner = Captioner()

    for fname in os.listdir(image_folder):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image = cv2.imread(os.path.join(image_folder, fname))
        start = time.time()
        text = captioner.caption(image)
        elapsed = time.time() - start
        print(fname, "-", text, "-", round(elapsed, 2), "seconds")