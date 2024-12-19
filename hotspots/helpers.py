import cv2
import numpy as np
from skimage.segmentation import expand_labels


def expand_labels_azran(labels, expand_distance, max_range):

    dsize = (2000, 2000)
    center = (dsize[0]/2, dsize[1]/2)
    radius = dsize[0]/2
    dx = max_range / radius

    expand_steps = int(expand_distance/dx)

    cart_2_polar_flag = cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP
    labels_cart = cv2.warpPolar(labels,
                                center=center,
                                maxRadius=radius,
                                dsize=dsize,
                                flags=cart_2_polar_flag)

    expanded_cart = expand_labels(labels_cart, expand_steps)

    expand_polar = cv2.warpPolar(expanded_cart,
                                 center=center,
                                 maxRadius=radius,
                                 dsize=(np.shape(labels)[1],
                                        np.shape(labels)[0]),
                                 flags=cv2.INTER_NEAREST)

    return expand_polar
