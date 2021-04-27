#!/usr/bin/python

import sys
import numpy as np
import skimage.io as skio
import skimage.color as skcolor
import skimage.util as skutil
import skimage.exposure as skexposure


def rgb2bn(img, alpha=0.5, beta=0.25):
    # BN = Blue - alpha*Red - beta*Green
    bn = img[..., 2] - alpha*img[..., 0] - beta*img[..., 1]
    np.maximum(bn, 0, out=bn) # clipping
    return bn


def get_n_classes_image_histogram(bn, num_intervals=10):
    hist, hist_centers = skexposure.histogram(bn)
    # Calculate revised histogram
    interval_len = len(hist) // num_intervals
    revised_hist = np.empty(num_intervals)
    revised_hist_centers = np.empty(num_intervals)
    for i in range(num_intervals):
        j = i * interval_len
        revised_hist[i] = np.max(hist[j: j+interval_len])
        revised_hist_centers[i] = hist_centers[j + np.argmax(hist[j: j+interval_len])]
    return revised_hist, revised_hist_centers


def find_valley(x, y):
    valley = x[-1]
    for i in range(1, len(x) - 1):
        if y[i-1] > y[i] and y[i] < y[i+1]:
            valley = x[i]
            break
    return valley


def get_brown_filter(img):
    bn = rgb2bn(img)
    bn = skutil.img_as_ubyte(bn) # [0.0, 1.0] -> [0, 255]

    revised_hist, revised_hist_centers = get_n_classes_image_histogram(bn)
    threshold = find_valley(revised_hist_centers, revised_hist)
    return (bn < threshold)[..., np.newaxis]


def get_blue_filter(de_dab):
    lab = skcolor.rgb2lab(de_dab)
    b = lab[..., 2] - np.min(lab[..., 2])

    revised_hist, revised_hist_centers = get_n_classes_image_histogram(b)
    # set the cut-off point before the peak
    threshold = revised_hist_centers[np.argmax(revised_hist) - 1]
    return (b < threshold)[..., np.newaxis]


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: " + sys.argv[0] + " input_image output_dab output_hematoxylin")
        sys.exit(1)

    tissue = skutil.img_as_float(skio.imread(sys.argv[1]))
    brown_filter = get_brown_filter(tissue)
    dab = tissue * brown_filter
    de_dab = tissue * (brown_filter == False)

    blue_filter = get_blue_filter(de_dab)
    hematoxylin = de_dab * blue_filter

    skio.imsave(sys.argv[2], skutil.img_as_ubyte(hematoxylin))
