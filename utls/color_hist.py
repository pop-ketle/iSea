# from: https://blog.shikoan.com/python-image-colorhist/
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def color_hist(filename):
    img = np.asarray(Image.open(filename).convert("RGB")).reshape(-1,3)
    plt.hist(img, color=["red", "green", "blue"], histtype="step", bins=128)
    plt.show()

color_hist('../datasets/satellite_images/2018/7Wc/7Wc_201812310000.png')