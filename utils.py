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
    im = im.squeeze()
    b, c, h, w = 0, 0, 0, 0
    if len(im.shape) > 4:
        raise ValueError("Image must be 2D or 3D")
    elif len(im.shape) == 4:
        # Batch of images
        b = im.shape[0]
        im = im[0]
    elif len(im.shape) == 3:
        # Color image or batch of monochrome images
        a, b, c = im.shape
        if a == 3:
            # Channel first
            b, c, h, w = 1, a, b, c
        elif c == 3:
            # Channel last
            b, c, h, w = 1, c, a, b
        elif a == 4:
            # Channel first, alpha coeff
            b, c, h, w = 1, a, b, c
        elif c == 4:
            # Channel last, alpha coeff
            b, c, h, w = 1, c, a, b
        else:
            # batch of monochrome images
            b, c, h, w = a, 1, b, c
    elif len(im.shape) == 2:
        # Monochrome image
        b, c, h, w = a, 1, b, c
    return im
