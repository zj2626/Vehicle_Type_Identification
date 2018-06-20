# coding=utf-8
# 改写自py-fast-rcnn中的demo

deep_root = "/home/zj/AAA___BIT-Vehicle/Vehicle-Detection/"

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from PIL import Image
from pylab import *

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
               'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, image_file, class_name, dets, app, thresh=0.5):
    bash_address = app.config['UPLOAD_FOLDER']
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        raise RuntimeError('error')
    im = im[:, :, (2, 1, 0)]

    fig, ax = plt.subplots(figsize=(round(im.shape[1] / 100, 2), round(im.shape[0] / 100, 2)))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        # draw the frame
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )

        left = int(bbox[0])
        top = int(bbox[1])
        right = int(bbox[2])
        bottom = int(bbox[3])
        img_default_name = image_file.split("/")[-1]
        cut_address = bash_address + _init_paths.save_cut_path \
                      + img_default_name.split(".")[0] + "_" + str(i) + "." + img_default_name.split(".")[1]
        print (cut_address)
        make_car_image(image_file, cut_address, left, top, right, bottom)

        # draw the name and score
        # ax.text(bbox[0], bbox[1] - 2,
        #         '{:s} {:.3f}'.format(class_name, score),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    full_address = bash_address + _init_paths.save_full_path + image_file.split("/")[-1]
    plt.savefig(full_address)

    return full_address


def make_car_image(img_file, new_file, left, top, right, bottom):
    img = Image.open(img_file)
    box = (left, top, right, bottom)
    img = img.crop(box)

    # show image
    # im = array(img)
    # imshow(im)

    img.save(new_file, 'jpeg')


def demo(net, image_file, app):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_file)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    scores, boxes = im_detect(net, im)

    full_address = ""
    cut_address = ""
    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        # only detect type: car
        if cls_ind != 6:
            continue

        # cls_ind -- > index  cls --> name
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        full_add = vis_detections(im, image_file, cls, dets, app, thresh=CONF_THRESH)
        full_address += full_add

    return full_address


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode', help='Use CPU mode (overrides --gpu)', action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]', choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args


def picture_detection(img, app):
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    timer = Timer()
    timer.tic()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print ('\n\nLoaded network {:s}'.format(caffemodel))

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(net, im)

    full_address = demo(net, img, app)
    timer.toc()

    return full_address, timer.total_time

    # im_names = ['000456.jpg', '000542.jpg', '001150.jpg']
    # for im_name in im_names:
    #     print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #     print ('图片地址 '.format(im_name))
    #     demo(net, im_name)

    # plt.show()
