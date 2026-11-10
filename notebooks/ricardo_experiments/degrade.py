import io
import random

import numpy as np
from PIL import Image, ImageFilter


class RandomDegradation:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        op = random.randrange(4)
        if op == 0:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=random.randint(30, 95))
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        if op == 1:
            return img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
        if op == 2:
            s = random.randint(16, 28)
            return img.resize((s, s), Image.BILINEAR).resize((32, 32), Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32)
        arr += np.random.randn(*arr.shape) * random.uniform(2, 10)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
