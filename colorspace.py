import math

import numpy as np
import torch

__all__ = ['RGBA_to_GRAY', 'RGBA_to_RGB', 'RGBA_to_CMYK', 'RGBA_to_HSV',
           'RGB_to_GRAY', 'RGB_to_HSV', 'RGB_to_CMYK', 'HSV_to_RGB', 'CMYK_to_RGB',
           'XYZ_to_LAB', 'RGB_to_XYZ',
           'GRAY_to_RGB']  # , 'RGB_to_LAB']

from matplotlib import colormaps as cm

from torch import Tensor


# -------- RGBA -----------------------#
class RGBA_to_GRAY:
    luma = {'SDTV': torch.Tensor([0.299, 0.587, 0.114, 1]),
            'Adobe': torch.Tensor([0.212, 0.701, 0.087, 1]),
            'HDTV': torch.Tensor([0.2126, 0.7152, 0.0722, 1]),
            'HDR': torch.Tensor([0.2627, 0.6780, 0.0593, 1])}

    def __init__(self):
        pass

    def __call__(self, im, luma: str = None, **kwargs):
        """
        Converts an RGBA image to grayscale using the specified luma coefficient or a default one.
        Warning : The Alpha transparency is lost during the operation
        Args:
            im (torch.Tensor): The input RGB image tensor.
            luma (str, optional): The name of the luma coefficient to use. Defaults to None.

        Returns:
            torch.Tensor: The grayscale image tensor.

        Raises:
            AssertionError: If the specified luma coefficient is not found in the dictionary.
        """
        assert im.colorspace == 'RGBA', "Wrong number of dimensions (/RGBA_to_GRAY)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if luma is not None:
            assert luma in self.luma
            im.data = torch.sum(torch.mul(im.permute(0, 2, 3, 1)[..., :-1], self.luma[luma]), dim=-1).unsqueeze(1)
        else:
            im.data = torch.sum(im[:, :-1, :, :] / 3, dim=1).unsqueeze(1)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='GRAY', num_ch=1)


class RGBA_to_RGB:

    def __init__(self):
        pass

    def __call__(self, im, **kwargs):
        """
        Converts an RGBA image to RGB.
        Warning : The Alpha transparency is lost during the operation
        Args:
            im (torch.Tensor): The input RGBA image tensor.

        Returns:
            torch.Tensor: The RGB image tensor.

        Raises:
            AssertionError: If the specified luma coefficient is not found in the dictionary.
        """
        assert im.colorspace == 'RGBA', "Starting Colorspace (/RGBA_to_RGB)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        im = im[:, :-1, :, :]
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='RGB', num_ch=3)


class RGBA_to_HSV:

    def __call__(self, im, luma: str = None, **kwargs):
        """
        Converts an RGBA image to the HSV colorspace.
        The alpha layer is lost !
        Args:
            im (torch.Tensor): The input RGBA image tensor.

        Returns:
            torch.Tensor: The HSV image tensor.
        """

        assert im.colorspace == 'RGBA', "Starting Colorspace (/RGBA_to_HSV)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)

        # ------- Hue ---------------- #
        R, G, B = Tensor(im[:, :1, :, :].data), Tensor(im[:, 1:2, :, :].data), Tensor(im[:, 2:3, :, :].data)
        Cmax, argCmax = torch.max(im, dim=1, keepdim=True)
        Cmin, _ = torch.min(im, dim=1, keepdim=True)
        Chroma = Cmax - Cmin
        Hue = torch.zeros_like(R, dtype=im.dtype)
        mask = Chroma == 0
        Hue[mask] = 0
        Hue[~mask & (argCmax == 0)] = 60 * (((G - B) / Chroma) % 6)[~mask & (argCmax == 0)]  # R is maximum
        Hue[~mask & (argCmax == 1)] = 60 * ((B - R) / Chroma + 2)[~mask & (argCmax == 1)]  # G is maximum
        Hue[~mask & (argCmax == 2)] = 60 * ((R - G) / Chroma + 4)[~mask & (argCmax == 2)]  # B is maximum
        # ------- Value ---------------- #
        Value, _ = torch.max(Tensor(im.data), dim=1, keepdim=True)
        # ------- Saturation ---------------- #
        Saturation = Value.clone()
        mask = Value != 0
        Saturation[mask] = Chroma[mask] / Value[mask]
        # ------- Stack the layers ----------- #
        im.data = torch.concatenate([Hue/360, Saturation, Value], dim=1)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='HSV', num_ch=3)


class RGBA_to_CMYK:

    def __call__(self, im, luma: str = None, **kwargs):
        """
        Converts an RGBA image to the HSV colorspace.

        Args:
            im (torch.Tensor): The input RGBA image tensor.

        Returns:
            torch.Tensor: The HSV image tensor.
        """

        assert im.colorspace == 'RGBA', "Starting Colorspace (/RGBA_to_HSV)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)

        R, G, B = Tensor(im[:, :1, :, :].data), Tensor(im[:, 1:2, :, :].data), Tensor(im[:, 2:3, :, :].data)
        # ------- Black Key K ---------------- #
        K = 1 - torch.max(im, dim=1, keepdim=True)[0]
        mask = K != 1
        # ------- Cyan ---------------- #
        C = torch.zeros_like(R, dtype=im.dtype)
        C[mask] = ((1 - R - K) / (1 - K))[mask]
        # ------- Magenta ---------------- #
        M = torch.zeros_like(R, dtype=im.dtype)
        M[mask] = ((1 - G - K) / (1 - K))[mask]
        # ------- Yellow ---------------- #
        Y = torch.zeros_like(R, dtype=im.dtype)
        Y[mask] = ((1 - B - K) / (1 - K))[mask]

        # ------- Stack the layers ----------- #
        im.data = torch.concatenate([C, M, Y, K], dim=1)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='CMYK', num_ch=4)

# -------- RGB -----------------------#
class RGB_to_GRAY:
    luma = {'SDTV': torch.Tensor([0.299, 0.587, 0.114, 1]),
            'Adobe': torch.Tensor([0.212, 0.701, 0.087, 1]),
            'HDTV': torch.Tensor([0.2126, 0.7152, 0.0722, 1]),
            'HDR': torch.Tensor([0.2627, 0.6780, 0.0593, 1])}

    def __init__(self):
        pass

    def __call__(self, im, luma: str = None, **kwargs):
        """
        Converts an RGB image to grayscale using the specified luma coefficient or a default one.

        Args:
            im (torch.Tensor): The input RGB image tensor.
            luma (str, optional): The name of the luma coefficient to use. Defaults to None.

        Returns:
            torch.Tensor: The grayscale image tensor.

        Raises:
            AssertionError: If the specified luma coefficient is not found in the dictionary.
        """
        assert im.colorspace == 'RGB', "Wrong number of dimensions (/RGB_to_GRAY)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if luma is not None:
            assert luma in self.luma
            im.data = torch.sum(torch.mul(im.permute(0, 2, 3, 1), self.luma[luma]), dim=-1).unsqueeze(1)
        else:
            im.data = torch.sum(im / 3, dim=1).unsqueeze(1)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='GRAY', num_ch=1, modality=0)


class RGB_to_HSV:

    def __call__(self, im, luma: str = None, **kwargs):
        """
        Converts an RGB image to the HSV colorspace.

        Args:
            im (torch.Tensor): The input RGB image tensor.

        Returns:
            torch.Tensor: The HSV image tensor.
        """

        assert im.colorspace == 'RGB', "Starting Colorspace (/RGB_to_HSV)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)

        # ------- Hue ---------------- #
        R, G, B = Tensor(im[:, :1, :, :].data), Tensor(im[:, 1:2, :, :].data), Tensor(im[:, 2:, :, :].data)
        Cmax, argCmax = torch.max(im, dim=1, keepdim=True)
        Cmin, _ = torch.min(im, dim=1, keepdim=True)
        Chroma = Cmax - Cmin
        Hue = torch.zeros_like(R, dtype=im.dtype)
        mask = Chroma == 0
        Hue[mask] = 0
        Hue[~mask & (argCmax == 0)] = 60 * (((G - B) / Chroma) % 6)[~mask & (argCmax == 0)]  # R is maximum
        Hue[~mask & (argCmax == 1)] = 60 * ((B - R) / Chroma + 2)[~mask & (argCmax == 1)]  # G is maximum
        Hue[~mask & (argCmax == 2)] = 60 * ((R - G) / Chroma + 4)[~mask & (argCmax == 2)]  # B is maximum
        # ------- Value ---------------- #
        Value, _ = torch.max(Tensor(im.data), dim=1, keepdim=True)
        # ------- Saturation ---------------- #
        Saturation = Value.clone()
        mask = Value != 0
        Saturation[mask] = Chroma[mask] / Value[mask]
        # ------- Stack the layers ----------- #
        im.data = torch.concatenate([Hue/360, Saturation, Value], dim=1)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='HSV', num_ch=3)

class RGB_to_CMYK:

    def __call__(self, im, luma: str = None, **kwargs):
        """
        Converts an RGB image to the HSV colorspace.

        Args:
            im (torch.Tensor): The input RGB image tensor.

        Returns:
            torch.Tensor: The HSV image tensor.
        """

        assert im.colorspace == 'RGB', "Starting Colorspace (/RGB_to_HSV)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)

        R, G, B = Tensor(im[:, :1, :, :].data), Tensor(im[:, 1:2, :, :].data), Tensor(im[:, 2:, :, :].data)
        # ------- Black Key K ---------------- #
        K = 1 - torch.max(im, dim=1, keepdim=True)[0]
        mask = K != 1
        # ------- Cyan ---------------- #
        C = torch.zeros_like(R, dtype=im.dtype)
        C[mask] = ((1 - R - K) / (1 - K))[mask]
        # ------- Magenta ---------------- #
        M = torch.zeros_like(R, dtype=im.dtype)
        M[mask] = ((1 - G - K) / (1 - K))[mask]
        # ------- Yellow ---------------- #
        Y = torch.zeros_like(R, dtype=im.dtype)
        Y[mask] = ((1 - B - K) / (1 - K))[mask]

        # ------- Stack the layers ----------- #
        im.data = torch.concatenate([C, M, Y, K], dim=1)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='CMYK', num_ch=4)

class RGB_to_XYZ:

    def __call__(self, im, luma: str = None, **kwargs):
        """
        Converts an RGB image to the LAB colorspace.

        Args:
            im (torch.Tensor): The input RGB image tensor.

        Returns:
            torch.Tensor: The LAB image tensor.
        """

        assert im.colorspace == 'RGB', "Starting Colorspace (/RGB_to_LAB)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)

        # ------- to XYZ ---------------- #
        mask = Tensor(im.data) > 0.04045
        temp = im.clone()
        temp[mask] = ((temp[mask] + 0.055) / 1.055) ** 2.4 * 100
        temp[~mask] = temp[~mask] / 12.92 * 100
        R, G, B = temp[:, :1, :, :], temp[:, 1:2, :, :], temp[:, 2:, :, :]

        # Observer= 2°, Illuminant= D65
        X = (R * 0.4124 + G * 0.3576 + B * 0.1805) / 95.047  # ref_X =  95.047
        Y = (R * 0.2126 + G * 0.7152 + B * 0.0722) / 100.0  # ref_Y = 100.000
        Z = (R * 0.0193 + G * 0.1192 + B * 0.9505) / 108.9  # ref_Z = 108.9

        # ------- Stack the layers ----------- #
        im.data = torch.concatenate([X, Y, Z], dim=1)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='XYZ', num_ch=3)


class XYZ_to_RGB:

    def __call__(self, im, luma: str = None, **kwargs):
        """
        Converts an XYZ image to the RGB colorspace.

        Args:
            im (torch.Tensor): The input XYZ image tensor.

        Returns:
            torch.Tensor: The RGB image tensor.
        """

        assert im.colorspace == 'XYZ', "Starting Colorspace (/XYZ_to_RGB)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)

        # ------- to XYZ ---------------- #
        mask = Tensor(im.data) > 0.04045
        temp = im.clone()
        temp[mask] = ((temp[mask] + 0.055) / 1.055) ** 2.4 * 100
        temp[~mask] = temp[~mask] / 12.92 * 100
        R, G, B = temp[:, :1, :, :], temp[:, 1:2, :, :], temp[:, 2:, :, :]

        # Observer= 2°, Illuminant= D65
        X = (R * 0.4124 + G * 0.3576 + B * 0.1805) / 95.047  # ref_X =  95.047
        Y = (R * 0.2126 + G * 0.7152 + B * 0.0722) / 100.0  # ref_Y = 100.000
        Z = (R * 0.0193 + G * 0.1192 + B * 0.9505) / 108.9  # ref_Z = 108.9

        # ------- Stack the layers ----------- #
        im.data = torch.concatenate([X, Y, Z], dim=1)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='XYZ', num_ch=3)


class XYZ_to_LAB:

    def __call__(self, im, luma: str = None, **kwargs):
        """
        Converts an XYZ image to the LAB colorspace.

        Args:
            im (torch.Tensor): The input XYZ image tensor.

        Returns:
            torch.Tensor: The LAB image tensor.
        """

        assert im.colorspace == 'XYZ', "Starting Colorspace (/XYZ_to_LAB)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        # ------- to LAB ---------------- #
        mask = Tensor(im.data) > (6/29)**3
        temp = im.clone()
        temp[mask] = temp[mask] ** (1/3)
        temp[~mask] = 1/3 * (29/6)**2 * temp[~mask] + 4/29
        X, Y, Z = temp[:, :1, :, :], temp[:, 1:2, :, :], temp[:, 2:, :, :]
        # ------- to LAB ---------------- #
        L = ((116 * Y) - 16) / 100
        a = (500 * (X - Y) + 128) / 256
        b = (200 * (Y - Z) + 128) / 256
        # ------- Stack the layers ----------- #
        im.data = torch.concatenate([L, a, b], dim=1)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='LAB', num_ch=3)


class LAB_to_XYZ:

    def __call__(self, im, luma: str = None, **kwargs):
        """
        Converts an XYZ image to the LAB colorspace.

        Args:
            im (torch.Tensor): The input XYZ image tensor.

        Returns:
            torch.Tensor: The LAB image tensor.
        """

        assert im.colorspace == 'LAB', "Starting Colorspace (/LAB_to_XYZ)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        temp = im.clone()
        L = (temp[:, :1, :, :] * 100 + 16) / 116
        a = (temp[:, 1:2, :, :] * 256 - 128) / 500
        b = (temp[:, 2:, :, :] * 256 - 128) / 200
        # ------- Y ---------------- #
        mask = L > (6/29)**3
        Y = torch.zeros_like(L)
        Y[mask] = L[mask] ** 3
        Y[~mask] = 3 * (6/29)**2 * (L[~mask] - 4/29)
        # ------- X ---------------- #
        mask = L + a > (6/29)**3
        X = L + a
        X[mask] = (X[mask]) ** 3
        X[~mask] = 3 * (6/29)**2 * (X[~mask] - 4/29)
        # ------- Z ---------------- #
        mask = L - b > (6/29)**3
        Z = L - b
        Z[mask] = (Z[mask]) ** 3
        Z[~mask] = 3 * (6/29)**2 * (Z[~mask] - 4/29)
        # ------- Stack the layers ----------- #
        im.data = torch.concatenate([X, Y, Z], dim=1)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='XYZ', num_ch=3)

# -------- HSV -----------------------#
class HSV_to_RGB:

    def __call__(self, im, luma: str = None, **kwargs):
        """
        Converts an HSV image to the RGB colorspace.

        Args:
            im (torch.Tensor): The input RGB image tensor.

        Returns:
            torch.Tensor: The HSV image tensor.
        """

        assert im.colorspace == 'HSV', "Starting Colorspace (/HSV_to_RGB)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        # ------- Intermediate layers ---------------- #
        H, S, V = Tensor(im[:, :1, :, :].data) * 359, Tensor(im[:, 1:2, :, :].data), Tensor(im[:, 2:, :, :].data)
        Chroma = V * S
        X = Chroma * (1 - torch.abs((H / 60) % 2 - 1))
        m = V - Chroma
        # ------- R, G, B ---------------- #
        R = torch.zeros_like(H, dtype=im.dtype)
        G = torch.zeros_like(H, dtype=im.dtype)
        B = torch.zeros_like(H, dtype=im.dtype)

        for a in range(6):
            angle = a * 60
            mask = ((H >= angle) * (H < (angle + 60)))
            if angle < 60:
                R[mask], G[mask], B[mask] = Chroma[mask], X[mask], 0
            elif angle < 120:
                R[mask], G[mask], B[mask] = X[mask], Chroma[mask], 0
            elif angle < 180:
                R[mask], G[mask], B[mask] = 0, Chroma[mask], X[mask]
            elif angle < 240:
                R[mask], G[mask], B[mask] = 0, X[mask], Chroma[mask]
            elif angle < 300:
                R[mask], G[mask], B[mask] = X[mask], 0, Chroma[mask]
            else:
                R[mask], G[mask], B[mask] = Chroma[mask], 0, X[mask]
        # ------- Stack the layers ----------- #
        im.data = torch.concatenate([R + m, G + m, B + m], dim=1)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='RGB', num_ch=3)

# -------- CMYK -----------------------#
class CMYK_to_RGB:

    def __call__(self, im, luma: str = None, **kwargs):
        """
        Converts an CMYK image to the RGB colorspace.

        Args:
            im (torch.Tensor): The input CMYK image tensor.

        Returns:
            torch.Tensor: The RGB image tensor.
        """

        assert im.colorspace == 'CMYK', "Starting Colorspace (/CMYK_to_RGB)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)

        C, M, Y, K = Tensor(im[:, :1, :, :].data), Tensor(im[:, 1:2, :, :].data), Tensor(im[:, 2:3, :, :].data), Tensor(im[:, 3:, :, :].data)
        # ------- R ---------------- #
        R = (1 - C) * (1 - K)
        # ------- G ---------------- #
        G = (1 - M) * (1 - K)
        # ------- B ---------------- #
        B = (1 - Y) * (1 - K)
        # ------- Stack the layers ----------- #
        im.data = torch.concatenate([R, G, B], dim=1)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='RGB', num_ch=3)

# -------- GRAY -----------------------#
class GRAY_to_RGB:

    def __call__(self, im, colormap='gray', **kwargs):
        """
        Converts an GRAY image to the RGB colorspace following a colormap.

        Args:
            im (torch.Tensor): The input GRAY image tensor.

        Returns:
            torch.Tensor: The RGB image tensor.
        """

        assert im.colorspace == 'GRAY', "Starting Colorspace (/GRAY_to_RGB)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if im.depth == 8:
            depth, datatype = 8, torch.uint8
        elif im.depth == 16:
            depth, datatype = 16, torch.uint32
        elif im.depth == 32:
            depth, datatype = 32, torch.uint32
        else:
            raise NotImplementedError
        L = Tensor(im[:, :, :, :].data)
        num = 2**(depth)
        x = np.linspace(0.0, 1.0, num)
        cmap_rgb = Tensor(cm[colormap](x)[:, :3]).to(im.device).squeeze()
        temp = (Tensor(im.data) * (num-1)).to(datatype).long().squeeze()
        new = cmap_rgb[temp].permute(2, 0, 1)
        # ------- Stack the layers ----------- #
        im.data = torch.concatenate([R, G, B], dim=1)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='RGB', num_ch=3)


def colorspace_fct(colorspace_change):
    if colorspace_change not in __all__:
        wrapper = __all__[f"{colorspace_change.split('_')[0]}_to_RGB"]
    else:
        wrapper = None
    if colorspace_change == 'RGBA_to_GRAY':
        fct = RGBA_to_GRAY()
    elif colorspace_change == 'RGBA_to_RGB':
        fct = RGBA_to_RGB()
    elif colorspace_change == 'RGBA_to_HSV':
        fct = RGBA_to_HSV()
    elif colorspace_change == 'RGBA_to_CMYK':
        fct = RGBA_to_CMYK()
    elif colorspace_change == 'RGB_to_GRAY':
        fct = RGB_to_GRAY()
    elif colorspace_change == 'RGB_to_HSV':
        fct = RGB_to_HSV()
    elif colorspace_change == 'RGB_to_CMYK':
        fct = RGB_to_CMYK()
    # elif colorspace_change == 'RGB_to_LAB':
    #     fct = RGB_to_LAB()
    elif colorspace_change == 'RGB_to_XYZ':
        fct = RGB_to_XYZ()
    elif colorspace_change == 'XYZ_to_LAB':
        fct = XYZ_to_LAB()
    elif colorspace_change == 'LAB_to_XYZ':
        fct = LAB_to_XYZ()
    elif colorspace_change == 'XYZ_to_RGB':
        fct = XYZ_to_RGB()
    elif colorspace_change == 'HSV_to_RGB':
        fct = HSV_to_RGB()
    elif colorspace_change == 'CMYK_to_RGB':
        fct = CMYK_to_RGB()
    elif colorspace_change == 'GRAY_to_RGB':
        fct = GRAY_to_RGB()
    else:
        raise NotImplementedError

    if wrapper is not None:
        fct = wrapper(fct)
    return fct
