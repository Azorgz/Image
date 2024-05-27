import os

import torch
# from colormaps import *
from Image import ImageTensor

vis = ImageTensor(os.getcwd() + "/vis.png")
vis = ImageTensor.rand(2, 3)
# vis.show()
vis2 = vis.GRAY()
# vis = vis2.LAB()
# vis.show()
vis2.show()
# RGBA = RGBA_to_GRAY()
