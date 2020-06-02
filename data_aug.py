import torch
import numpy as np

def get_mask(mask, mask_annotations):
    for regions in annotations['regions']:
        x = regions['shape_attributes']['all_points_x']
        y = regions['shape_attributes']['all_points_y']

        rr, cc = skimage.draw.polygon(y, x)
        mask[rr, cc] = 1
    return mask