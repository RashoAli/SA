import numpy as np
import matplotlib.pyplot as plt
import cv2

gfg = np.array([[[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 10], [11, 12]]])
print(np.shape(gfg))
print(np.transpose((gfg), (1, 2, 0)))
