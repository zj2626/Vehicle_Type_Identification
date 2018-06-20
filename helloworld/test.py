# coding=utf-8

from skimage import transform, data
import matplotlib.pyplot as plt
#
# img = data.camera()
# dst = transform.resize(img, (80, 60))
# plt.figure('resize')
#
# plt.subplot(1, 2, 1)
# plt.title('before resize')
# plt.imshow(img)
#
# plt.subplot(1, 2, 2)
# plt.title('after im_detectresize')
# plt.imshow(dst, plt.cm.gray)
#
# plt.show()

# import _init_paths
# from fast_rcnn.test import im_detect
# help(im_detect)

from utils.timer import Timer
help(Timer)