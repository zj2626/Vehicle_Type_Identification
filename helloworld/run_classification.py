# coding=utf-8

deep_root = "/home/zj/Documents/deepfile/BIT-Vehicle/"

net_root = deep_root + "train_data_with_position_full/"
caffe_model = deep_root + "8.model_caffe_50000_new_data_full/caffenet_train_iter_50000.caffemodel"
net_file = net_root + "caffe_deploy.prototxt"
mean_file = net_root + 'mean.npy'

net_root2 = deep_root + "train_data_with_position/"
caffe_model2 = deep_root + "5.model_caffe_60000_new_data/caffenet_train_iter_60000.caffemodel"
net_file2 = net_root2 + "caffe_deploy.prototxt"
mean_file2 = net_root + 'mean.npy'

import numpy as np

np.set_printoptions(suppress=True)

import _init_paths
from utils.timer import Timer
import caffe

labels_filename = deep_root + "labels.txt"


def picture_classification(img, ifFUllImage):
    timer = Timer()
    timer.tic()

    net, transformer = create_net(ifFUllImage)
    # 正式加载图片
    img = caffe.io.load_image(img)

    # 用上面的transformer.preprocess来处理刚刚加载图片 然后把数据传入data层
    net.blobs['data'].data[...] = transformer.preprocess('data', img)

    # 网络开始向前传播
    out = net.forward()

    data = net.blobs['prob'].data[0].flatten()
    # print (data)
    timer.toc()

    # 加载label信息
    imagenet_labels_filename = labels_filename
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

    # flatten返回把numpy对象折叠成一维的数组 argsort将其中的元素从小到大排列并提取其对应的索引
    top_index = data.argsort()[-1:-6:-1]
    # top_l = labels[top_index[0]]
    # top_p = data[top_index[0]]

    top = []

    # 当前这个图片的属于哪个物体的概率
    for i in np.arange(top_index.size):
        if top_index[i] < 6:
            print (top_index[i], labels[top_index[i]], data[top_index[i]])
            det = {'label': labels[top_index[i]], 'prob': round(data[top_index[i]] * 100, 4)}
            top.append(det)

    return top, timer.total_time


def create_net(ifFUllImage):
    # 处理图片 把上面添加的两个变量都作为参数构造一个Net
    net = caffe.Net(net_file2 if ifFUllImage else net_file,
                    caffe_model2 if ifFUllImage else caffe_model,
                    caffe.TEST)
    # 得到data的形状，这里的图片是默认matplotlib底层加载的
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # matplotlib加载的image是像素[0-1],图片的数据格式[weight,high,channels]，RGB
    # caffe加载的图片需要的是[0-255]像素，数据格式[channels,weight,high],BGR，那么就需要转换
    # channel 放到前面
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(mean_file2 if ifFUllImage else mean_file).mean(1).mean(1))
    # 图片像素放大到[0-255]
    transformer.set_raw_scale('data', 255)
    # RGB-->BGR 转换
    transformer.set_channel_swap('data', (2, 1, 0))
    # 改变输入data的维度，使根据图片大小
    # net.blobs['data'].reshape(1, 3, 227, 227)

    return net, transformer
