# coding=utf-8
# !/usr/bin/env python

"""Set up paths for Fast R-CNN."""
import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_root = osp.join('/home/zj/caffe/python')
add_path(caffe_root)

# Add lib to PYTHONPATH
lib_path = osp.join('/home/zj/py-faster-rcnn/lib')
add_path(lib_path)

save_full_path = "full_"
save_cut_path = "cut_"