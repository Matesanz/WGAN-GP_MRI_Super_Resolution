import numpy as np
from utils.minc_viewer import Viewer
from cv2 import compareHist, HISTCMP_BHATTACHARYYA
from skimage.metrics import structural_similarity as ssim


def mse(x, y):
        return round(np.linalg.norm(x - y), 4)

def ssim_3d(x, y):
    ssim_2d = [ssim(xj, yj) for xj, yj in zip(x,y)]
    return round(np.mean(np.asarray(ssim_2d)), 4)

def battacharyya(x, y):
    bat = [compareHist(xj, yj, HISTCMP_BHATTACHARYYA) for xj, yj in zip(x,y)]
    return round(np.mean(np.asarray(bat)), 4)
