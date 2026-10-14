import os
import time

import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

model_name = "Salesforce/blip-image-captioning-base"


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
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # num_beams=1 is the fastest. try 3 later to see if captions get better
            out = self.model.generate(**inputs, max_new_tokens=30, num_beams=1)
        return self.processor.decode(out[0], skip_special_tokens=True)


# running this file directly captions every image in test_images (your old test)
if __name__ == "__main__":
    image_folder = "test_images"
    captioner = Captioner()

    for fname in os.listdir(image_folder):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image = cv2.imread(os.path.join(image_folder, fname))
        start = time.time()
        text = captioner.caption(image)
        elapsed = time.time() - start
        print(fname, "-", text, "-", round(elapsed, 2), "seconds")