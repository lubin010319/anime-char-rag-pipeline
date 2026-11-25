"""One WD14 CPU provider and one image/text encoder."""
import csv

import numpy as np
from PIL import Image

from common import ALIASES, JA

WD_MODEL = "SmilingWolf/wd-v1-4-vit-tagger-v2"
WD_REVISION = "1f3f3e8ae769634e31e1ef696df11ec37493e4f2"
EMBED_MODEL = "google/siglip2-base-patch16-224"
EMBED_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"


class WDTagger:
    def __init__(self, threshold=0.30):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            hf_hub_download(WD_MODEL, "model.onnx", revision=WD_REVISION), sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input = self.session.get_inputs()[0]
        with open(hf_hub_download(WD_MODEL, "selected_tags.csv", revision=WD_REVISION), encoding="utf-8", newline="") as file:
            self.labels = list(csv.DictReader(file))
        self.threshold = threshold

    def __call__(self, image):
        side = max(image.size)
        padded = Image.new("RGB", (side, side), "white")
        padded.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        size = self.input.shape[1]
        rgb = np.asarray(padded.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32)
        # The model consumes BGR NHWC values on a 0–255 scale.
        bgr = np.ascontiguousarray(rgb[..., ::-1][None])
        values = self.session.run(None, {self.input.name: bgr})[0][0]
        return {row["name"]: round(float(score), 4) for row, score in zip(self.labels, values, strict=True)
                if row["category"] == "0" and score >= self.threshold}


def character_tags(tags):
    return sorted({ALIASES.get(tag, tag) for tag in tags if ALIASES.get(tag, tag) in JA})


class Embedder:
    def __init__(self, device, revision=EMBED_REVISION):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device == "cpu":
            torch.set_num_threads(min(torch.get_num_threads(), 6))
        self.processor = AutoProcessor.from_pretrained(EMBED_MODEL, revision=revision, use_fast=False)
        self.model = AutoModel.from_pretrained(EMBED_MODEL, revision=revision).to(self.device).eval()
        self.revision = self.model.config._commit_hash or revision

    def vector(self, *, images=None, texts=None):
        if (images is None) == (texts is None):
            raise ValueError("Supply exactly one modality")
        with self.torch.inference_mode():
            if images is not None:
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                features = self.model.get_image_features(**inputs)
            else:
                inputs = self.processor(text=texts, padding="max_length", max_length=64,
                                        truncation=True, return_tensors="pt").to(self.device)
                features = self.model.get_text_features(**inputs)
            output = self.torch.nn.functional.normalize(features.float(), dim=-1)
        return np.ascontiguousarray(output.cpu().numpy())
