# Session 13 — captcha digit images → resize, normalize, LabelBinarizer.
# Digit folders live in kaptcha/1 … kaptcha/9 next to this file.
# From the repo root: uv run python sessions/session13/kaptcha_sample.py
#
# CRISP
#
# Business Understanding
# Data Understanding
#
# X => unstructured data => Images
# y => Folder Names
#
# pip install opencv-python
# built-in package

from pathlib import Path

import cv2 as cv
import glob

_SESSION = Path(__file__).resolve().parent
_KAPTCHA = _SESSION / "kaptcha"


def load_data(dataset: str):
    images = []
    labels = []

    for item in glob.glob(dataset):
        img = cv.imread(item)
        # Data Preparation
        # resize
        img = cv.resize(img, (32, 32))  # Conformed متحدالشکل
        # normalize
        # img = img / 255.0
        img = cv.normalize(
            img,
            None,
            alpha=0,
            beta=1,
            norm_type=cv.NORM_MINMAX,
            dtype=cv.CV_32FC3,
        )
        images.append(img)
        label = Path(item).parent.name
        labels.append(label)

    return images, labels


images, labels = load_data(str(_KAPTCHA / "*" / "*"))
print(len(images), len(labels))

# LLM
# OCR
# [ 1 0 0 0 0 0 0 0 0 ] => 1
# [ 0 0 0 0 0 0 0 0 1 ] => 9

# pip install scikit-learn
# pip install numpy

from sklearn.preprocessing import LabelBinarizer
import numpy as np

# Numpy Array
images = np.array(images)

lb = LabelBinarizer()

labels = lb.fit_transform(labels)
print(images.shape, labels.shape)
# print(labels)
# print(images)

















