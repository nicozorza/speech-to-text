import numpy as np
from scipy.misc import toimage


def expand_image(image, expand_factor):
    new_image = []
    start = True
    for row in image:
        new_list = []
        for item in row:
            new_list += np.repeat(item, expand_factor).tolist()
        stacked_list = new_list
        for i in range(expand_factor - 1):
            stacked_list = np.row_stack((stacked_list, new_list))

        if start:
            new_image = stacked_list
            start = False
        else:
            new_image = np.row_stack((new_image, stacked_list))

    return new_image


def save_alignment_image(image, filename, expand_factor=0):
    image = expand_image(image, expand_factor) if expand_factor > 0 else image
    new_image = 255 - (image * 255)
    im = toimage(new_image)
    im.save(filename)


