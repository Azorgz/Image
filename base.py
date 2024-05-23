from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from warnings import warn

import numpy as np
from torch import Tensor

"""
Base classes for ImageTensor, largely inspired from Kornia Image files
https://github.com/kornia/kornia/blob/main/kornia/image/base.py
"""

dict_modality = {'Any': 0, 'Visible': 1, 'Multimodal': 2}
mode_list = np.array(['UNKNOWN', 'BINARY', 'GRAY', 'RGB', 'RGBA', 'CMYK', 'LAB', 'HSV'])
mode_dict = {'UNKNOWN': 0, 'BINARY': 1, 'GRAY': 2, 'RGB': 3,
             'RGBA': 4, 'CMYK': 5, 'LAB': 6, 'HSV': 7}


@dataclass(frozen=True)
class Modality(Enum):
    r"""Data class to represent image modality.
    modality, either :
    Visible (3,4 channels)
    Multimodal (2 + channels)
    Any (1 channel lengthwave)"""
    Any = 0
    Visible = 1
    Multimodal = 2


@dataclass()
class ImageSize:
    r"""Data class to represent image shape.

    Args:
        height: image height.
        width: image width.

    Example:
        >>> size = ImageSize(3, 4)
        >>> size.height
        3
        >>> size.width
        4
    """

    height: int | Tensor
    width: int | Tensor


class ColorSpace(Enum):
    r"""Enum that represents the color space of an image."""

    UNKNOWN = 0  # in case of multi band images
    BINARY = 1  # in case of binary mask images (1, 3, or 4 channel)
    GRAY = 2  # in case of grayscale images (1 channel)
    RGB = 3  # in case of color images (3 channel)
    RGBA = 4  # in case of color images (4 channel)
    CMYK = 5  # in case of color images in CMYK mode (4 channel)
    LAB = 6  # in case of color images in LAB mode (3 channel)
    HSV = 7  # in case of color images in HSV mode (3 channel)


@dataclass()
class PixelFormat:
    r"""Data class to represent the pixel format of an image.

    Args:
        colorspace: color space.
        bit_depth: the number of bits per channel.

    Example:
        >>> pixel_format = PixelFormat(ColorSpace.RGB, 8)
        >>> pixel_format.colorspace
        <ColorSpace.RGB: 2>
        >>> pixel_format.bit_depth
        8
    """
    colorspace: ColorSpace
    bit_depth: int


@dataclass()
class Channel:
    r"""Enum that represents the channels order of an image."""
    pos: int
    num_ch: int


@dataclass()
class Dims:
    r"""list that represents the dims order of an image."""
    batch: int = 0
    channels: int = 1
    height: int = 2
    width: int = 3

    def permute(self, dims):
        temp = np.array(self.dims)[dims]
        self.batch, self.channels, self.height, self.width = (temp.tolist())

    @property
    def dims(self):
        return [self.batch, self.channels, self.height, self.width]

    @property
    def layers(self):
        return np.array(['batch', 'channels', 'height', 'width'])[self.dims].tolist()


@dataclass()
class Batch:
    r"""Enum that represents the batch dimension."""
    batched: bool
    batch_size: int


@dataclass()
class ImageLayout:
    """Data class to represent the layout of an image.
    """
    modality: Modality
    image_size: ImageSize
    channel: Channel
    pixel_format: PixelFormat
    channel_names = ['']
    batch: Batch
    dims: Dims

    def __init__(self, modality: Modality,
                 image_size: ImageSize,
                 channel: Channel,
                 pixel_format: PixelFormat,
                 batch: Batch,
                 channel_names: list = None,
                 dims: list = None):
        self.modality = modality
        self.image_size = image_size
        self.channel = channel
        self.pixel_format = pixel_format
        self.batch = batch
        if dims is not None:
            self.dims = Dims(*dims)
        else:
            self.dims = Dims()
        if channel_names is not None:
            try:
                assert len(channel_names) == channel.num_ch
                self.channel_names = channel_names
            except AssertionError:
                warn("The given channel names don't match the number of channels")
                self.channel_names = None
        else:
            self.channel_names = None

        self._CHECK_MODALITY_VALIDITY()
        self._CHECK_CHANNEL_VALIDITY()
        self._CHECK_COLOR_VALIDITY()
        self._CHECK_DEPTH_VALIDITY()
        self._CHECK_LAYERS_VALIDITY()

    def __eq__(self, other):
        return self.modality == other.modality and \
            self.image_size == other.image_size and \
            self.channel == other.channel and \
            self.pixel_format == other.pixel_format and \
            self.channel_names == other.channel_names and \
            self.batch == other.batch and \
            self.dims == other.dims

    def clone(self):
        return ImageLayout(modality=Modality(self.modality.value),
                           image_size=ImageSize(self.image_size.height, self.image_size.width),
                           channel=Channel(self.channel.pos, self.channel.num_ch),
                           pixel_format=PixelFormat(self.pixel_format.colorspace, self.pixel_format.bit_depth),
                           channel_names=self.channel_names,
                           batch=Batch(self.batch.batched, self.batch.batch_size),
                           dims=self.dims.dims)

    def _CHECK_MODALITY_VALIDITY(self):
        if self.modality.name == 'Visible':
            assert self.channel.num_ch in [3, 4]
        elif self.modality.name == 'Multimodal':
            assert self.channel.num_ch > 1
        else:
            assert self.channel.num_ch == 1

    def _CHECK_LAYERS_VALIDITY(self):
        assert self.dims.batch != self.dims.channels != self.dims.height != self.dims.width
        assert int(np.argwhere(np.array(self.dims.dims) == 1)) == self.channel.pos

    def _CHECK_CHANNEL_VALIDITY(self):
        assert self.channel.pos in [0, 1, 2, 3]
        assert self.channel.num_ch >= 1

    def _CHECK_COLOR_VALIDITY(self):
        cs = self.pixel_format.colorspace
        if cs == 2:  # GRAY
            assert self.channel.num_ch == 1
        elif cs in [3, 6, 7]:
            assert self.channel.num_ch == 3
        elif cs in [4, 5]:
            assert self.channel.num_ch == 4
        elif cs == 1:  # Mask image
            assert self.channel.num_ch in [1, 3, 4]

    def _CHECK_DEPTH_VALIDITY(self):
        assert self.pixel_format.bit_depth in [8, 16, 32, 64]

    def _CHECK_BATCH_SIZE(self):
        if self.batch.batched:
            assert self.batch.batch_size > 1

    def update(self, **kwargs):
        if 'height' in kwargs or 'width' in kwargs or 'image_size' in kwargs:
            h = kwargs['height'] if 'height' in kwargs else self.image_size.height
            w = kwargs['width'] if 'width' in kwargs else self.image_size.width
            h, w = (kwargs['image_size'][0], kwargs['image_size'][1]) if 'image_size' in kwargs else (h, w)
            self._update_image_size(height=h, width=w)
        if 'pos' in kwargs or 'num_ch' in kwargs:
            pos = kwargs['pos'] if 'pos' in kwargs else self.channel.pos
            num_ch = kwargs['num_ch'] if 'num_ch' in kwargs else self.channel.num_ch
            self._update_channel(pos=pos, num_ch=num_ch)
        if 'colorspace' in kwargs or 'bit_depth' in kwargs:
            cs = kwargs['colorspace'] if 'colorspace' in kwargs else self.pixel_format.colorspace
            if not isinstance(cs, ColorSpace):
                assert cs in mode_dict, 'The required colospace is not implemented (/update layout)'
                cs = ColorSpace(mode_dict[cs])
            bd = kwargs['bit_depth'] if 'bit_depth' in kwargs else self.pixel_format.bit_depth
            self._update_pixel_format(colorspace=cs, bit_depth=bd)
        if 'batch_size' in kwargs:
            batched = kwargs['batch_size'] > 1
            batch_size = kwargs['batch_size']
            self._update_batch(batched=batched, batch_size=batch_size)
        if 'channel_names' in kwargs:
            self._update_channel_names(kwargs['channel_names'])
        if 'dims' in kwargs:
            self._update_dims(kwargs['dims'])

    def _update_channel(self, **kwargs):
        self.channel = Channel(**kwargs)
        self._CHECK_CHANNEL_VALIDITY()

    def _update_dims(self, dims):
        try:
            assert len(dims) == 4
            self.dims.permute(dims)
        except AssertionError:
            warn("The given channel dims don't match the number of dimensions")

    def _update_channel_names(self, names):
        try:
            assert len(names) == self.channel.num_ch
            self.channel_names = names
        except AssertionError:
            warn("The given channel names don't match the number of channels")
            self.channel_names = None

    def _update_image_size(self, **kwargs):
        self.image_size = ImageSize(**kwargs)

    def _update_pixel_format(self, **kwargs):
        self.pixel_format = PixelFormat(**kwargs)
        self._CHECK_COLOR_VALIDITY()
        self._CHECK_CHANNEL_VALIDITY()
        self._CHECK_DEPTH_VALIDITY()

    def _update_batch(self, **kwargs):
        self.batch = Batch(**kwargs)
        self._CHECK_BATCH_SIZE()
