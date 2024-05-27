from __future__ import annotations

import imp
import os
import warnings
from collections.abc import Iterable
from os.path import *
from typing import Union
from warnings import warn

import PIL.Image
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import colormaps as cm
from matplotlib import pyplot as plt, patches
from torch import Tensor
# from torchvision.transforms.functional import to_pil_image

from base import Modality, ColorSpace, Channel, ImageLayout, dict_modality, mode_list
from colorspace import colorspace_fct
from utils import update_channel_pos, find_class, find_best_grid, CHECK_IMAGE_SHAPE, \
    CHECK_IMAGE_FORMAT, CHECK_LAYOUT, pil_to_numpy


class ImageTensor(Tensor):
    """
    A class defining the general basic framework of a TensorImage.
    The modality per default is VIS (visible light), but can be multimodal or LWIR, SWIR...
    It can use all the methods from Torch plus some new ones.
    To create a new instance:
    --> From file (a str or path pointing towards an image file)
    --> From numpy array (shape h, w, c / h, w, c, a or h, w in case of mono-layered image)
    --> From a torch tensor (shape c, h, w  or b, c, h, w for batched tensor)
    An instance is created using a numpy array or a path to an image file or a PIL image
    """
    _image_layout: ImageLayout = None
    _modality: str = None
    _im_name: str = 'new image'
    _im_pad = torch.tensor([[0, 0, 0, 0]])
    _colorspace: str = None
    _mode_list = mode_list
    _pixel_depth: int = None
    _channel_pos: int | Tensor = None
    _channel_num: int | Tensor = None
    _channelNames: list = None
    _layer_name: list = None
    _batched: bool = False

    def __init__(self, *args, **kwargs):
        super(ImageTensor, self).__init__()

    # ------- Instance creation methods ---------------------------- #

    @staticmethod
    def __new__(cls, inp, *args,
                name: str = None,
                device: torch.device = None,
                modality: str = None,
                colorspace: str = None,
                channel_names=None,
                image_layout=None,
                **kwargs):
        # Input array is a path to an image OR an already formed ndarray instance
        if isinstance(image_layout, ImageLayout):
            assert CHECK_LAYOUT(*CHECK_IMAGE_SHAPE(inp), image_layout)
            inp_ = inp
        else:
            if isinstance(inp, str):
                name = basename(inp).split('.')[0] if name is None else name
                inp = Image.open(inp)
                inp = pil_to_numpy(inp)
                pad = torch.tensor([[0, 0, 0, 0]])
            if isinstance(inp, ImageTensor):
                inp_ = Tensor(inp)
                image_layout = inp.image_layout.clone()
                name = str(inp.im_name)
                pad = inp.im_pad
            elif isinstance(inp, np.ndarray) or isinstance(inp, Tensor):
                valid, inp_, image_size, channel, batch = CHECK_IMAGE_SHAPE(inp)
                if colorspace is not None:
                    colorspace = int(np.argwhere(mode_list == colorspace)[0][0])
                inp_, pixelformat, mod = CHECK_IMAGE_FORMAT(inp_, colorspace)
                modality = mod if modality is None else Modality(dict_modality[modality])
                image_layout = ImageLayout(modality, image_size, channel, pixelformat, batch,
                                           channel_names=channel_names)
                pad = torch.tensor([[0, 0, 0, 0]])
            else:
                raise NotImplementedError

        if isinstance(device, torch.device):
            inp_ = inp_.to(device)
        image = super().__new__(cls, inp_)
        # add the new attributes to the created instance of Image
        image._image_layout = image_layout
        image._im_name = name
        image._im_pad = pad
        return image

    @classmethod
    def rand(cls, batch: int = 1, channels: int = 3, height: int = 100, width: int = 100,
             depth: int | str = 32):
        dtype_dict = {str(32): torch.float32, str(64): torch.float64}
        assert str(depth) in dtype_dict, 'depth must be either 32 or 64 bits'
        dtype = dtype_dict[str(depth)]
        assert channels >= 1 and batch >= 1 and height >= 1 and width >= 1
        return cls(torch.rand([batch, channels, height, width], dtype=dtype), name='Random Image')

    @classmethod
    def randint(cls, batch: int = 1, channels: int = 3, height: int = 100, width: int = 100,
                depth: int | str = 8):
        dtype_dict = {str(8): torch.uint8, str(16): torch.uint16, str(32): torch.uint32}
        assert str(depth) in dtype_dict, 'depth must be either 8, 16 or 32 bits'
        dtype = dtype_dict[str(depth)]
        assert channels >= 1 and batch >= 1 and height >= 1 and width >= 1
        if dtype == torch.uint8:
            high = 255
        elif dtype == torch.uint16:
            high = 65535
        elif dtype == torch.uint32:
            high = 4294967295
        else:
            high = 18446744073709551615
        return cls(torch.randint(0, high, [batch, channels, height, width], dtype=dtype), name='Random Image')

    # ------- Torch function call method ---------------------------- #

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # print(f"Calling '{func.__name__}' for Subclass")
        return super().__torch_function__(func, types, args=args, kwargs=kwargs)
        res = super().__torch_function__(func, types, args=args, kwargs=kwargs)
        if res.__class__ is Tensor:
            res = ImageTensor(res)
        if res.__class__ is ImageTensor:
            arg = find_class(args, ImageTensor)
            if arg is not None:
                res.pass_attr(arg)
                if res.shape != arg.shape:
                    res.channel_pos = abs(update_channel_pos(res))
            return res
        else:
            return res

    def pass_attr(self, image, *args):
        if len(args) > 0:
            for arg in args:
                if arg == 'image_layout':
                    self.__dict__[arg] = image.__dict__[arg].clone()
                self.__dict__[arg] = image.__dict__[arg].copy()
        else:
            self.image_layout = image.image_layout.clone()
            self.im_name = image.im_name
            self.im_pad = image.im_pad

    def __eq__(self, other):
        if isinstance(other, ImageTensor):
            eq = True
            if torch.sum(Tensor(self.data) - Tensor(other.data)) != 0:
                eq = False
            elif self.image_layout != other.image_layout:
                eq = False
            return eq
        else:
            return Tensor(self.data) == other

    def clone(self, *args):
        """
        Function to clone an ImageTensor
        :param image: ImageTensor to clone
        :return: cloned ImageTensor
        """
        new = ImageTensor(self)
        return new

    @torch.no_grad()
    def batch(self, images: list):
        """
        Function to batch ImageTensor together
        :param images: list of ImageTensor to batch
        :return: batched ImageTensor
        NO GRAD
        """
        assert isinstance(images, list)
        assert len(images) > 0
        batch = self.batch_size
        for i, im in enumerate(images):
            assert isinstance(im, ImageTensor), 'Only ImageTensors are supported'
            if im.shape != self.shape:
                images[i] = Tensor(im.match_shape(self))
            batch += im.batch_size
        new = ImageTensor(torch.concatenate([self.data, *images], dim=0))
        new.batch_size = batch
        return new

    @torch.no_grad()
    def interpolate(self, items):
        """
        :param items:  List/array or Tensor of shape (N,2) of tuple of coordinate to interpolate
        :return: Tensor of interpolated values
        """
        try:
            device = items.device
        except AttributeError:
            device = 'cpu'
        h, w = self.image_size
        N = self.batch_size
        grid = Tensor(items).unsqueeze(-1).repeat(N, 1, 1, 1)
        grid[:, :, :, 0] = grid[:, :, :, 0] * 2 / w - 1
        grid[:, :, :, 1] = grid[:, :, :, 1] * 2 / h - 1
        return F.grid_sample(Tensor(self.to(device)), grid, align_corners=True).squeeze()

    # Image manipulation methods // Size changes //
    def pad(self, im, **kwargs):
        '''
        Pad the image to match the given Tensor/Array size or with the list of padding indicated (left, right, top, bottom)
        :param im:
        :param kwargs:
        :return: a copy of self but padded
        '''
        if isinstance(im, ImageTensor):
            h, w = im.image_size
        elif isinstance(im, Tensor) or isinstance(im, np.ndarray):
            h, w = im.shape[-2:]
        elif isinstance(im, list) or isinstance(im, tuple):
            assert len(im) == 2 or len(im) == 4
            if len(im) == 2:
                pad_l, pad_r = int(im[0]), int(im[0])
                pad_t, pad_b = int(im[1]), int(im[1])
            elif len(im) == 4:
                pad_l, pad_r, pad_t, pad_b = int(im[0]), int(im[1]), int(im[2]), int(im[3])
            else:
                pad_l, pad_r, pad_t, pad_b = 0, 0, 0, 0
            pad_tuple = (pad_l, pad_r, pad_t, pad_b)
            new = F.pad(self, pad_tuple, **kwargs)
            new.image_size = new.shape[-2:]
            return new
        else:
            h, w = 0, 0
        h_ref, w_ref = self.image_size
        try:
            assert w >= w_ref and h >= h_ref
        except AssertionError:
            return self.clone()
        pad_l = int((w - w_ref) // 2 + (w - w_ref) % 2)
        pad_r = int((w - w_ref) // 2 - (w - w_ref) % 2)
        pad_t = int((h - h_ref) // 2 + (h - h_ref) % 2)
        pad_b = int((h - h_ref) // 2 - (h - h_ref) % 2)
        pad_tuple = (pad_l, pad_r, pad_t, pad_b)

        new = F.pad(self, pad_tuple, **kwargs)
        new.image_size = new.shape[-2:]
        return new

    def hstack(self, *args, **kwargs):
        assert all([im.image_layout == self.image_layout for im in args])
        temp = [Tensor(self.put_channel_at(-1).permute([1, 2, 0, 3]))]
        for im in args:
            temp.append(Tensor(im.put_channel_at(-1).permute([1, 2, 0, 3])))
        temp = torch.hstack(temp).permute([2, 3, 0, 1])
        res = self.clone()
        res.data = temp
        res.image_size = res.shape[-2:]
        res.channel_pos = 1
        return res

    def vstack(self, *args, **kwargs):
        assert all([im.image_layout == self.image_layout for im in args])
        temp = [Tensor(self.put_channel_at(-1).permute([1, 2, 0, 3]))]
        for im in args:
            temp.append(Tensor(im.put_channel_at(-1).permute([1, 2, 0, 3])))
        temp = torch.vstack(temp).permute([2, 3, 0, 1])
        res = self.clone()
        res.data = temp
        res.image_size = res.shape[-2:]
        res.channel_pos = 1
        return res

    def pyrDown(self):
        out = self.put_channel_at(1)
        # downsample
        out.data = F.interpolate(Tensor(out.data),
                                 scale_factor=1 / 2,
                                 mode='bilinear',
                                 align_corners=True)
        out.image_size = out.shape[-2:]
        out.put_channel_at(self.channel_pos)
        return out

    def pyrUp(self):
        out = self.put_channel_at(1)
        # upsample
        out.data = F.interpolate(
            Tensor(out.data),
            scale_factor=2,
            mode='bilinear',
            align_corners=True)
        out.image_size = out.shape[-2:]
        out.put_channel_at(self.channel_pos)
        return out

    def reset_layers_order(self, in_place: bool = True):
        dims = np.array(self.image_layout.dims.dims)
        new_dims = [int(np.argwhere(dims == i)) for i in range(len(dims))]
        if in_place:
            self.permute(new_dims, in_place=True)
        else:
            return self.permute(new_dims, in_place=False)

    def put_channel_at(self, idx=1, in_place: bool = False):
        if in_place:
            if idx == self.channel_pos:
                return self
            self.data = torch.movedim(Tensor(self.data), self.channel_pos, idx)
            self.channel_pos = idx
        else:
            new = self.clone()
            if idx == self.channel_pos:
                return new
            new.data = torch.movedim(Tensor(self.data), self.channel_pos, idx)
            new.channel_pos = idx
            return new

    def permute(self, dims, in_place: bool = False):
        """
        Similar to permute torch function but with Layers tracking.
        Work with a list of : layer indexes, layer names ('batch', 'height', 'width', 'channels')
        :param in_place: bool to state if a new tensor has to be generated or not
        :param dims: List of new dimension order (length = 4)
        :return: ImageTensor or Nothing
        """
        dims = dims.copy()
        if any(isinstance(d, str) for d in dims):
            layers = np.array(self.layers_name)
            for idx, d in enumerate(dims):
                if isinstance(d, str):
                    if d == 'batch' or d == 'b':
                        dims[idx] = int(np.argwhere(layers == 'batch'))
                    elif d == 'height' or d == 'h':
                        dims[idx] = int(np.argwhere(layers == 'height'))
                    elif d == 'width' or d == 'w':
                        dims[idx] = int(np.argwhere(layers == 'width'))
                    elif d == 'channels' or d == 'c' or d == 'channel':
                        dims[idx] = int(np.argwhere(layers == 'channels'))
                    else:
                        raise ValueError(f'Unknown dimension {d}')
        assert len(np.unique(dims)) == len(dims), 'Dimension position must be unique (/permute)'
        new_channel_pos = int(np.argwhere((np.array(dims) == self.channel_pos)))
        if in_place:
            temp = self
        else:
            temp = self.clone()
        temp.data = torch.permute(Tensor(temp.data), dims)
        temp.channel_pos = new_channel_pos
        temp.layers_name = dims
        if not in_place:
            return temp

    def match_shape(self, other: Union[Tensor, tuple, list], keep_ratio=False):
        """
        Take as input either a Tensor based object to match on size or
        an Iterable describing the new size to get
        :param other: Tensor like or Iterable
        :param keep_ratio: to match on size while keeping the original ratio
        :return: ImageTensor
        """
        temp = self.put_channel_at()
        if isinstance(other, tuple) or isinstance(other, list):
            shape = other
            assert len(other) == 2
        elif isinstance(other, ImageTensor) or isinstance(other, DepthTensor):
            shape = other.image_size
        else:
            shape = other.shape[-2:]
        if keep_ratio:
            ratio = torch.tensor(self.image_size) / torch.tensor(shape)
            ratio = ratio.max()
            temp.data = F.interpolate(temp.data, mode='bilinear', scale_factor=float((1 / ratio).cpu().numpy()))
            temp = temp.pad(other)
        else:
            temp.data = F.interpolate(temp.data, size=shape, mode='bilinear', align_corners=True)
        temp.image_size = temp.shape[-2:]
        return temp.put_channel_at(self.channel_pos)

    def resize(self, shape, keep_ratio=False):
        temp = self.put_channel_at()
        if keep_ratio:
            ratio = torch.tensor(self.image_size) / torch.tensor(shape)
            ratio = ratio.max()
            temp.data = F.interpolate(Tensor(temp.data), mode='bilinear', scale_factor=float(1 / ratio))
            temp.image_size = temp.shape[-2:]
        else:
            temp.data = F.interpolate(Tensor(temp.data), size=shape, mode='bilinear', align_corners=True)
            temp.image_size = temp.shape[-2:]
        temp.put_channel_at(self.channel_pos, in_place=True)
        return temp

    def squeeze(self, *args, **kwargs):
        return Tensor(self.data).squeeze(*args, **kwargs)

    def unsqueeze(self, *args, **kwargs):
        return Tensor(self.data).unsqueeze(*args, **kwargs)

    def normalize(self, return_minmax=False, keep_abs_max=False):
        if keep_abs_max:
            a = torch.abs(self.data)
            m = a.min()
            M = a.max()
            a = (a - m) / (M - m)
        else:
            m = self.min()
            M = self.max()
            a = self.clone()
            a.data = (a.data - m) / (M - m)
        if return_minmax:
            return a, m, M
        else:
            return a

    # utils methods
    def to_opencv(self, datatype=np.uint8):
        """
        :return: np.ndarray
        """
        if self.modality == 'Visible':
            return self.permute(['b', 'h', 'w', 'c']).to_numpy(datatype=datatype)[..., [2, 1, 0]].squeeze()
        else:
            return self.permute(['b', 'h', 'w', 'c']).to_numpy(datatype=datatype).squeeze()

    def to_numpy(self, datatype=np.uint8):
        """
        :param datatype: np.uint8, np.uint16, np.float32, np.float64
        :return: np.ndarray
        """
        if datatype == np.uint8:
            factor = 255
        elif datatype == np.uint16:
            factor = 256 ** 2 - 1
        else:
            factor = 1
        if self.requires_grad:
            numpy_array = np.ascontiguousarray(Tensor.numpy((self * factor).detach().cpu()), dtype=datatype)
        else:
            numpy_array = np.ascontiguousarray(Tensor.numpy((self * factor).cpu()), dtype=datatype)
        return numpy_array

    @torch.no_grad()
    def show(self, num=None, cmap='gray', roi: list = None, point: Union[list, Tensor] = None):
        im = self.permute(['b', 'h', 'w', 'c'], in_place=False)
        # If the ImageTensor is multimodal or batched then we will plot a matrix of images for each mod / image
        if im.modality == 'Multimodal' or im.batch_size > 1:
            im._multiple_show(num=num, cmap=cmap)
        # Else we will plot a Grayscale image or a ColorImage
        else:
            if num is None:
                num = self.im_name
            if self.channel_names is not None:
                num += ' / ' + self.channel_names[0]
            fig, ax = plt.subplots(ncols=1, nrows=1, num=num, squeeze=False)
            if im.modality == 'Any':
                im = im.RGB('gray')
            else:
                cmap = None
            im_display = im.to_numpy().squeeze()
            if point is not None:
                for center in point.squeeze():
                    center = center.cpu().long().numpy()
                    im_display = cv.circle(im_display[..., [2, 1, 0]], center, 5, (0, 255, 0), -1)[..., [2, 1, 0]]
            ax[0, 0].imshow(im_display, cmap=cmap)
            if roi is not None:
                for r, color in zip(roi, ['r', 'g', 'b']):
                    rect = patches.Rectangle((r[1], r[0]), r[3] - r[1], r[2] - r[0]
                                             , linewidth=2, edgecolor=color, facecolor='none')
                    ax[0, 0].add_patch(rect)
            ax[0, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            plt.show()
            return ax[0, 0]

    def _multiple_show(self, num=None, cmap='gray'):

        if self.modality != 'Visible':
            im_display = self.to_numpy()
            im_display = [*im_display.reshape([self.batch_size * self.channel_num, *self.image_size])]
        else:
            im_display = self.permute(['b', 'h', 'w', 'c']).to_numpy()
            im_display = [*im_display]
        if not num:
            num = self.im_name
        rows, cols = find_best_grid(len(im_display))
        fig = plt.figure(num=num)
        axes = [(fig.add_subplot(rows, cols, r * cols + c + 1) if (r * cols + c + 1 <= len(im_display)) else None) for r
                in range(rows) for c in range(cols)]
        for i, img in enumerate(im_display):
            if self.modality != 'Any':
                cmap = None
            axes[i].imshow(img, cmap=cmap)
        for a in axes:
            if a is not None:
                a.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()
        return axes

    def save(self, path, name=None, ext='png', **kwargs):
        name = self.im_name + f'.{ext}' if name is None else name
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if not cv.imwrite(path + f'/{name}', self.opencv()):
            raise Exception("Could not write image")

    def to_tensor(self):
        """
        Remove all attributes to keep only the data as a torch tensor.
        :return: Tensor
        """
        return torch.Tensor(self.data)

    # ---------------- Properties -------------------------------- #

    @property
    def im_name(self) -> str:
        return self._im_name

    @im_name.setter
    def im_name(self, name) -> None:
        self._im_name = name

    @property
    def image_layout(self) -> ImageLayout:
        return self._image_layout

    @image_layout.setter
    def image_layout(self, value) -> None:
        self._image_layout = value

    @property
    def im_pad(self):
        return self._im_pad

    @im_pad.setter
    def im_pad(self, value) -> None:
        self._im_pad = value

    # ---------------- Inherited from layout -------------------------------- #
    @property
    def batched(self) -> bool:
        return self.image_layout.batch.batched

    @property
    def batch_size(self) -> int:
        return self.image_layout.batch.batch_size

    @batch_size.setter
    def batch_size(self, value):
        self.image_layout.update(batch_size=value)

    @property
    def image_size(self) -> tuple:
        return self.image_layout.image_size.height, self.image_layout.image_size.width

    @image_size.setter
    def image_size(self, value):
        self.image_layout.update(image_size=value)

    @property
    def depth(self) -> int:
        return self.image_layout.pixel_format.bit_depth

    @depth.setter
    def depth(self, value) -> None:
        self.image_layout.update(bit_depth=value)

    @property
    def modality(self) -> str:
        return self.image_layout.modality.name

    @property
    def mode_list(self) -> list:
        return self._mode_list

    @property
    def layers_name(self) -> list:
        return self.image_layout.dims.layers

    @layers_name.setter
    def layers_name(self, dims) -> None:
        self.image_layout.update(dims=dims)

    @property
    def channel_pos(self) -> int:
        return self.image_layout.channel.pos

    @channel_pos.setter
    def channel_pos(self, pos) -> None:
        self.image_layout.update(pos=pos if pos >= 0 else pos + self.ndim)

    @property
    def channel_names(self) -> list:
        return self.image_layout.channel_names

    @channel_names.setter
    def channel_names(self, names) -> None:
        self.image_layout.update(channel_names=names)

    @property
    def channel_num(self) -> int:
        return self.image_layout.channel.num_ch

    @channel_num.setter
    def channel_num(self, num) -> None:
        self.image_layout.update(num_ch=num)

    @property
    def colorspace(self) -> str:
        return self.image_layout.pixel_format.colorspace.name

    @colorspace.setter
    def colorspace(self, v) -> None:
        """
        :param c_mode: str following the Modes of a Pillow Image
        :param colormap: to convert a GRAYSCALE image to a Palette (=colormap) colored image
        """
        if self.modality == 'Multimodal':
            warnings.warn('Multimodal Images cannot get a colorspace')
            return
        if isinstance(v, list) or isinstance(v, tuple):
            colorspace = v[0]
            colormap = v[1]['colormap'] if v[1] else 'inferno'
        else:
            colorspace = v
            colormap = 'inferno'
        if colorspace == self.colorspace:
            return
        elif self.colorspace == 'BINARY':
            warnings.warn("The Mask image can't be colored")
            return
        else:
            colorspace_change_fct = colorspace_fct(f'{self.colorspace}_to_{colorspace}')
            colorspace_change_fct(self, colormap=colormap)

        #     x = np.linspace(0.0, 1.0, 256)
        #     # cmap_rgb = Tensor(cm.get_cmap(plt.get_cmap(colormap))(x)[:, :3]).to(self.device).squeeze()
        #     cmap_rgb = Tensor(cm[colormap](x)[:, :3]).to(self.device).squeeze()
        #     temp = (self * 255).long().squeeze()
        #     new = ImageTensor(cmap_rgb[temp].permute(2, 0, 1), color_mode='RGB')
        #     self.data = new.data

    # ---------------- Colorspace change functions -------------------------------- #
    def RGB(self, cmap='gray'):
        """
        Implementation equivalent at the attribute setting : im.colorspace = 'rgb' but create a new ImageTensor
        """
        im = self.clone()
        im.colorspace = 'RGB', {'colormap': cmap}
        return im

    def RGBA(self, cmap='gray'):
        """
        Implementation equivalent at the attribute setting : im.colorspace = 'rgba' but create a new ImageTensor
        """
        im = self.clone()
        im.colorspace = 'RGBA', {'colormap': cmap}
        return im

    def GRAY(self):
        """
        Implementation equivalent at the attribute setting : im.colorspace = 'gray' but create a new ImageTensor
        """
        im = self.clone()
        im.colorspace = 'GRAY', {}
        return im

    def CMYK(self, cmap='gray'):
        """
        Implementation equivalent at the attribute setting : im.colorspace = 'cmyk' but create a new ImageTensor
        """
        im = self.clone()
        im.colorspace = 'CMYK', {'colormap': cmap}
        return im

    def LAB(self, cmap='gray'):
        """
        Implementation equivalent at the attribute setting : im.colorspace = 'lab' but create a new ImageTensor
        """
        im = self.clone()
        im.colorspace = 'LAB', {'colormap': cmap}
        return im

    def HSV(self, cmap='gray'):
        """
        Implementation equivalent at the attribute setting : im.colorspace = 'hsv' but create a new ImageTensor
        """
        im = self.clone()
        im.colorspace = 'HSV', {'colormap': cmap}
        return im

    def XYZ(self, cmap='gray'):
        """
        Implementation equivalent at the attribute setting : im.colorspace = 'xyz' but create a new ImageTensor
        """
        im = self.clone()
        im.colorspace = 'XYZ', {'colormap': cmap}
        return im

    def BINARY(self):
        """
        Implementation equivalent at the attribute setting : im.colorspace = '1' but create a new ImageTensor
        """
        im = self.clone()
        im.colorspace = 'BINARY', {'threshold': 0.5},
        return im


class DepthTensor(ImageTensor):
    """
    A SubClass of Image Tensor to deal with Disparity/Depth value > 1.
    If the Tensor is modified, the maximum value always be referenced
    """
    _max_value = 0
    _min_value = 0
    _scaled = False
    _ori_shape = None
    _mode_list = ['L', 'RGB']
    _im_type = 'Depth'
    _color_mode = 'L'

    @staticmethod
    def __new__(cls, im: Union[ImageTensor, Tensor], device: torch.device = None):
        inp = im.squeeze()
        assert len(inp.shape) == 2 or len(inp.shape) == 3
        max_value = inp.max()
        min_value = inp.min()
        inp_ = (inp - min_value) / (max_value - min_value)
        inp_ = super().__new__(cls, inp_, device=device)
        if len(inp.shape) == 3:
            inp_ = inp_.unsqueeze(1)
        inp_._max_value = max_value
        inp_._min_value = min_value
        inp_._ori_shape = inp_.shape[-2:]
        inp_.im_type = 'Depth'
        return inp_

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # print(f"Calling '{func.__name__}' for Subclass")
        res = super().__torch_function__(func, types, args=args, kwargs=kwargs)
        if res.__class__ is Tensor:
            res = DepthTensor(res).scale()
        if res.__class__ is DepthTensor:
            arg = find_class(args, DepthTensor)
            if arg is not None:
                res.pass_attr(arg)
                if res.shape != arg.shape:
                    res.channel_pos = abs(update_channel_pos(res))
            return res
        else:
            return res

    def show(self, num=None, cmap='inferno', roi: list = None, point: Union[list, Tensor] = None):
        im_display = [*self]
        if not num:
            num = self.im_name
        fig, ax = plt.subplots(ncols=len(im_display), num=num, squeeze=False)
        for i, img in enumerate(im_display):
            im_display = img.squeeze()
            # im_display = (im_display - im_display.min()) / (im_display.max() - im_display.min())
            if len(im_display.shape) > 2:
                im_display, cmap = im_display.permute(1, 2, 0), None
            else:
                im_display, cmap = im_display, cmap
            if point is not None:
                for center in point.squeeze():
                    center = center.cpu().long().numpy()
                    im_display = cv.circle(im_display.opencv(), center, 5, (0, 255, 0), -1)[..., [2, 1, 0]]
            ax[0, i].imshow(im_display.detach().cpu().numpy(), cmap=cmap)
            if roi is not None:
                for r, color in zip(roi, ['r', 'g', 'b']):
                    rect = patches.Rectangle((r[1], r[0]), r[3] - r[1], r[2] - r[0]
                                             , linewidth=2, edgecolor=color, facecolor='none')
                    ax[0, i].add_patch(rect)

            ax[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()
        return ax

    def save(self, path, name=None, save_image=False, **kwargs):
        if save_image:
            ImageTensor(self.inverse_depth(remove_zeros=True)
                        .normalize()).RGB().save(path, name=name)
        else:
            name = self.im_name + '.tiff' if name is None else name
            im = self.opencv()
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            if not cv.imwrite(path + f'/{name}', im):
                raise Exception("Could not write image")

    def opencv(self, **kwargs):
        if self.color_mode == 'L':
            a = np.ascontiguousarray(Tensor.numpy(self.unscale().squeeze().cpu()) * 255 ** 2, dtype=np.uint16)
        else:
            a = np.ascontiguousarray(
                Tensor.numpy(self.unscale().put_channel_at(-1).squeeze().cpu())[..., [2, 1, 0]] * 255 ** 2,
                dtype=np.uint16)
        return a

    def clamp(self, mini=None, maxi=None, *, out=None):
        self.max_value = min(maxi, self.max_value)
        self.min_value = max(mini, self.min_value)
        return torch.clamp(self, min=mini, max=maxi)

    def normalize(self, minmax=False, keep_abs_max=True):
        a = super().normalize(minmax=minmax, keep_abs_max=keep_abs_max)
        return a

    @property
    def scaled(self):
        return self._scaled

    @scaled.setter
    def scaled(self, value):
        self._scaled = value

    def scale(self):
        new = self.clone()
        if not new.scaled:
            new.scaled = True
            return new * (new.max_value - new.min_value) + new.min_value
        else:
            return new

    def unscale(self):
        new = self.clone()
        if new.scaled:
            new.scaled = False
            return (new - new.min_value) / (new.max_value - new.min_value)
        else:
            return new

    def inverse_depth(self, remove_zeros=False, remove_max=True, factor=100):
        temp = self.clone()
        if remove_zeros:
            temp[temp == 0] = temp.max()
        if remove_max:
            temp[temp == temp.max()] = temp.min()
        temp = factor / (temp + 1)
        return temp.normalize()

    @property
    def color_mode(self) -> str:
        return self._color_mode

    @color_mode.setter
    def color_mode(self, v) -> None:
        pass

    @property
    def max_value(self):
        return self._max_value

    @max_value.setter
    def max_value(self, v):
        self._max_value = v

    @property
    def min_value(self):
        return self._min_value

    @min_value.setter
    def min_value(self, v):
        self._min_value = v

    @property
    def ori_shape(self):
        return self._ori_shape
