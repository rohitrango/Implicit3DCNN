import numpy as np

def z_score_normalize(image):
    return (image - image.mean()) / image.std()

def uniform_normalize(image):
    return (image - image.min()) / (image.max() - image.min()) * 2 - 1
