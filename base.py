from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from torch import Tensor

"""
Base classes for ImageTensor, largely inspired from Kornia Image files
https://github.com/kornia/kornia/blob/main/kornia/image/base.py
"""


@dataclass(frozen=True)
class Modality:
    r"""Data class to represent image modality.
    modality, either :
    Visible (3,4 channels)
    Multimodal (2 + channels)
    Any (1 channel lengthwave)"""
    mod: str = 'Any'


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
        color_space: color space.
        bit_depth: the number of bits per channel.

    Example:
        >>> pixel_format = PixelFormat(ColorSpace.RGB, 8)
        >>> pixel_format.color_space
        <ColorSpace.RGB: 2>
        >>> pixel_format.bit_depth
        8
    """
    color_space: ColorSpace
    bit_depth: int


@dataclass()
class Channel:
    r"""Enum that represents the channels order of an image."""
    pos: int
    num_ch: int


@dataclass()
class ImageLayout:
    """Data class to represent the layout of an image.
    """
    modality: Modality
    image_size: ImageSize
    channel: Channel
    pixel_format: PixelFormat

    def __init__(self, modality: Modality,
                 image_size: ImageSize,
                 channel: Channel,
                 pixel_format: PixelFormat):
        self.modality = modality
        self.image_size = image_size
        self.channel = channel
        self.pixel_format = pixel_format

        self._CHECK_MODALITY_VALIDITY()
        self._CHECK_CHANNEL_VALIDITY()
        self._CHECK_COLOR_VALIDITY()
        self._CHECK_DEPTH_VALIDITY()

    def _CHECK_MODALITY_VALIDITY(self):
        assert self.modality.mod in ['Visible', 'Multimodal', 'Any']
        if self.modality.mod == 'Visible':
            assert self.channel.num_ch in [3, 4]
        elif self.modality.mod == 'Multimodal':
            assert self.channel.num_ch > 1
        else:
            assert self.channel.num_ch == 1

    def _CHECK_CHANNEL_VALIDITY(self):
        assert self.channel.pos in [1, -1]
        assert self.channel.num_ch in [1, 3, 4]

    def _CHECK_COLOR_VALIDITY(self):
        cs = self.pixel_format.color_space
        if cs == 2:  # GRAY
            assert self.channel.num_ch == 1
        elif cs in [3, 6, 7]:
            assert self.channel.num_ch == 3
        elif cs in [4, 5]:
            assert self.channel.num_ch == 4
        elif cs == 1:  # Mask image
            assert self.channel.num_ch in [1, 3, 4]

    def _CHECK_DEPTH_VALIDITY(self):
        assert self.pixel_format.bit_depth in [8, 16, 32]

    def update(self, **kwargs):
        if 'height' in kwargs or 'width' in kwargs:
            h = kwargs['height'] if 'height' in kwargs else self.image_size.height
            w = kwargs['width'] if 'width' in kwargs else self.image_size.width
            self._update_image_size(height=h, width=w)
        if 'pos' in kwargs or 'num_ch' in kwargs:
            pos = kwargs['pos'] if 'pos' in kwargs else self.channel.pos
            num_ch = kwargs['num_ch'] if 'num_ch' in kwargs else self.channel.num_ch
            self._update_channel(pos=pos, num_ch=num_ch)
        if 'color_space' in kwargs or 'bit_depth' in kwargs:
            cs = kwargs['color_space'] if 'color_space' in kwargs else self.pixel_format.color_space
            bd = kwargs['bit_depth'] if 'bit_depth' in kwargs else self.pixel_format.bit_depth
            self._update_pixel_format(color_space=cs, bit_depth=bd)

    def _update_channel(self, **kwargs):
        self.channel = Channel(**kwargs)
        self._CHECK_CHANNEL_VALIDITY()

    def _update_image_size(self, **kwargs):
        self.image_size = ImageSize(**kwargs)

    def _update_pixel_format(self, **kwargs):
        self.pixel_format = PixelFormat(**kwargs)
        self._CHECK_COLOR_VALIDITY()
        self._CHECK_CHANNEL_VALIDITY()
        self._CHECK_DEPTH_VALIDITY()
