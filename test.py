import torch
from colormaps import *
from Image import ImageTensor

vis = ImageTensor("/home/godeta/PycharmProjects/Image/vis.png")
# vis.show()
vis2 = vis.HSV()
vis2 = vis2.RGB()
vis.show()
vis2.show()
# RGBA = RGBA_to_GRAY()
