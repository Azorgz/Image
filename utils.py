import numpy as np
import torch
import PIL.Image
from PIL import Image
from torch import Tensor
from base import ImageSize, Channel, ColorSpace, PixelFormat, Modality, Batch, ImageLayout


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


def CHECK_LAYOUT(valid:bool, inp: Tensor, image_size:ImageSize, channel:Channel, batch:Batch, layout:ImageLayout):
    assert valid, 'Shape is not valid'
    assert image_size == layout.image_size, 'The size of the given tensor doesnt match the size of the given Layout'
    assert channel == layout.channel, 'The number and position of channels of the given Tensor doesnt match the Layout'
    assert batch == layout.batch, 'The batch of the given Tensor doesnt match the Layout'
    return True


def CHECK_IMAGE_SHAPE(im: np.ndarray | Tensor | PIL.Image.Image):
    """
    Return first a boolean to indicate whether the image shape is valid or not
    Return the image with channels at the right positions,
    the ImageSize, the Channel and the Batch
    :param im: image to check
    :return: channels order
    """
    if isinstance(im, Image.Image):
        im = pil_to_numpy(im)
    im0 = im.squeeze()
    b = -1
    valid = True
    if len(im0.shape) > 4:
        return False

    elif len(im0.shape) == 4:
        b_ = 1
        b = 0
        # Batch of images
        im0 = im0[0]
    else:
        b_ = 0

    if len(im0.shape) == 3:
        # Color image or batch of monochrome images
        c, h, w = im0.shape
        if c == 3:
            # Channel first
            c, h, w = 0 + b_, 1 + b_, 2 + b_
        elif w == 3:
            # Channel last
            c, h, w = 2 + b_, 0 + b_, 1 + b_
        elif c == 4:
            # Channel first, alpha coeff
            c, h, w = 0 + b_, 1 + b_, 2 + b_
        elif w == 4:
            # Channel last, alpha coeff
            c, h, w = 2 + b_, 0 + b_, 1 + b_
        else:
            # batch of monochrome images or batch of multimodal images
            # channel position : smallest dimension
            c = int(np.argmin(np.array([c, h, w])) + b_)
            l = [b_, 1 + b_, 2 + b_]
            l.remove(c)
            h, w = l[0], l[1]

    elif len(im0.shape) == 2:
        # Monochrome image
        b, c, h, w = -1, -2, 0, 1
    else:
        # Unknown image shape
        return False

    if isinstance(im, np.ndarray):
        while len(im.shape) < 4:
            im = np.expand_dims(im, axis=-1)
        im = im.transpose([b, c, h, w])
        im = torch.from_numpy(im)
    elif isinstance(im, torch.Tensor):
        im = im.permute([b, c, h, w])
    return (valid,
            im,
            ImageSize(*im.shape[-2:]),
            Channel(1, im.shape[1]),
            Batch(im.shape[0] > 1, im.shape[0]))


def CHECK_IMAGE_FORMAT(im, colorspace):
    # Depth format
    if im.dtype == torch.uint8:
        bit_depth = 8
        im = im / 255
    elif im.dtype == torch.uint16:
        bit_depth = 16
        im = im / (256 ** 2 - 1)
    elif im.dtype == torch.float32 or im.dtype == torch.uint32:
        bit_depth = 32
        if im.dtype == torch.uint32:
            im = im / (256 ** 4 - 1)
    elif im.dtype == torch.float64:
        bit_depth = 64
    elif im.dtype == torch.bool:
        bit_depth = 1
    else:
        raise NotImplementedError
    if im.max() > 1 or im.min() < 0:
        im = (im - im.min()) / (im.max() - im.min())
    # Colorspace check
    c = im.shape[1]
    if colorspace is None:
        if bit_depth == 1:
            colorspace = ColorSpace(1)
            modality = 0
        elif c == 1:
            colorspace = ColorSpace(2)
            modality = 0
        elif c == 3:
            colorspace = ColorSpace(3)
            modality = 1
        elif c == 4:
            colorspace = ColorSpace(4)
            modality = 1
        else:
            colorspace = ColorSpace(0)
            modality = 2
    else:
        if bit_depth == 1:
            modality = 0
        elif c == 1:
            modality = 0
        elif c == 3:
            modality = 1
        elif c == 4:
            modality = 1
        else:
            modality = 2
    return im, PixelFormat(colorspace, bit_depth), Modality(modality)





