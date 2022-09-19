import os
import cv2
import numpy as np
from PIL import Image


def _to_pil(img_buf):
    if isinstance(img_buf, bytes):
        img = np.asarray(bytearray(img_buf), dtype='uint8')
        image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif isinstance(img_buf, str) and os.path.exists(img_buf):
        image = Image.open(img_buf)
        image = image.convert('RGB')
    elif isinstance(img_buf, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(img_buf, cv2.COLOR_BGR2RGB))
    elif isinstance(img_buf, Image.Image):
        image = img_buf
    else:
        image = None
    
    return image