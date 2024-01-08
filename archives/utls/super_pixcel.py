# ../datasets/satellite_images/2020/7Wc/7Wc_202001010000.png


import cv2
import numpy as np

def main():
    image = cv2.imread('../datasets/satellite_images/2020/7Wc/7Wc_202001010000.png')

    # パラメータ
    height, width, channels= image.shape[:3]
    num_iterations = 10
    prior = 2
    double_step = True
    num_superpixels = 700
    num_levels = 4
    num_histogram_bins = 5

    # スーパーピクセルSEEDSの生成
    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels,
            num_levels, prior, num_histogram_bins, double_step)

    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

    # 画像のスーパーピクセルSEEDSを計算
    seeds.iterate(converted, num_iterations)

    # スーパーピクセルセグメンテーションの境界を取得
    contour_mask = seeds.getLabelContourMask(False)
    result = image.copy()
    result[0 < contour_mask] = (0, 255, 255)

    # 画像表示
    cv2.imshow('SEEDS result', result)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
