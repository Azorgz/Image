import numpy as np
import torch
from PIL.Image import Image


def find_class(args, class_name):
    arg = {}
    for a in args:
        if isinstance(a, class_name):
            return a
        elif isinstance(a, list) or isinstance(a, tuple):
            arg = find_class(a, class_name)
            if arg != {}:
                return arg
        else:
            arg = None
    return arg


def update_channel_pos(im):
    shape = np.array(im.shape)
    channel_pos = np.argwhere(shape == 3)
    channel_pos = channel_pos[0][0] if len(channel_pos >= 1) else \
        (np.argwhere(shape == 1)[0][0] if len(np.argwhere(shape == 1)) >= 1 else None)
    if channel_pos is None:
        return -1
    else:
        return int(channel_pos)


def pil_to_numpy(im):
    """
    Converts a PIL Image object to a NumPy array.
    Source : Fast import of Pillow images to NumPy / OpenCV arrays Written by Alex Karpinsky

    Args:
        im (PIL.Image.Image): The input PIL Image object.

    Returns:
        numpy.ndarray: The NumPy array representing the image.
    """
    im.load()

    # Unpack data
    e = Image._getencoder(im.mode, "raw", im.mode)
    e.setimage(im.im)

    # NumPy buffer for the result
    shape, typestr = Image._conv_type_shape(im)
    data = np.empty(shape, dtype=np.dtype(typestr))
    mem = data.data.cast("B", (data.data.nbytes,))

    bufsize, s, offset = 65536, 0, 0
    while not s:
        l, s, d = e.encode(bufsize)

        mem[offset:offset + len(d)] = d
        offset += len(d)
    if s < 0:
        raise RuntimeError("encoder error %d in tobytes" % s)
    return data


def find_best_grid(param):
    srt = int(np.floor(np.sqrt(param)))
    i = 0
    while srt * (srt + i) < param:
        i += 1
    return srt, srt + i


def CHECK_IMAGE_SHAPE(im):
    """
    Return the position of each channel in this order b, c, h, w
    :param im: image to check
    :return: channels order
    """
    im0 = im.squeeze()
    b, c, h, w = 0, 0, 0, 0

    if len(im0.shape) > 4:
        raise ValueError("Image must be 2D or 3D")

    elif len(im0.shape) == 4:
        b = 1
        # Batch of images
        im0 = im0[0]

    if len(im0.shape) == 3:
        # Color image or batch of monochrome images
        c, h, w = im0.shape
        if c == 3:
            # Channel first
            c, h, w = 0 + b, 1 + b, 2 + b
        elif w == 3:
            # Channel last
            c, h, w = 1 + b, 2 + b, 0 + b
        elif c == 4:
            # Channel first, alpha coeff
            c, h, w = 0 + b, 1 + b, 2 + b
        elif w == 4:
            # Channel last, alpha coeff
            c, h, w = 1 + b, 2 + b, 0 + b
        else:
            # batch of monochrome images
            c, h, w = -1, 1, 2

    elif len(im0.shape) == 2:
        # Monochrome image
        b, c, h, w = -1, -1, 0, 1

    while len(im.shape) < 4:
        im = im.unsqueeze(-1)
    if isinstance(im, np.ndarray):
        im = np.transpose([b, c, h, w])
    elif isinstance(im, Image):
        im = np.transpose([b, c, h, w])
    elif isinstance(im, torch.Tensor):
        im = im.permute([b, c, h, w])
    return im
