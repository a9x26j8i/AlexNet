# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import math
import os


def get_files(file_dir, ratio):
    '''
    参数：
     file_dir: trainval根目录
     ratio:    train，validation 比例
    返回:
     图像和标签list
    '''
    hege = []
    label_hege = []
    bie = []
    label_bie = []
    huang = []
    label_huang = []
    ji = []
    label_ji = []
    shuang = []
    label_shuang = []
    doc_list = os.listdir(file_dir)
    for doc in doc_list:
        name_doc = doc.split(sep='_')[0]   #检查文件夹名字，确定类别
        if name_doc == 'hege':
            label = 0
            hege, label_hege = make_namelist_lablelist(file_dir, doc, label)
            print('There are %d hege canjian'%(len(hege)))
        elif name_doc == 'bie':
            label = 1
            bie, label_bie = make_namelist_lablelist(file_dir, doc, label)
            print('There are %d bie canjian' % (len(bie)))
        elif name_doc == 'huang':
            label = 2
            huang, label_huang = make_namelist_lablelist(file_dir, doc, label)
            print('There are %d huang canjian' % (len(huang)))
        elif name_doc == 'ji':
            label = 3
            ji, label_ji = make_namelist_lablelist(file_dir, doc, label)
            print('There are %d ji canjian' % (len(ji)))
        elif name_doc == 'shuang':
            label = 4
            shuang, label_shuang = make_namelist_lablelist(file_dir, doc, label)
            print('There are %d shuang canjian' % (len(shuang)))

    image_list = np.hstack((hege, bie, huang, ji, shuang))
    label_list = np.hstack((label_hege, label_bie, label_huang, label_ji, label_shuang))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]

    n_sample = len(all_image_list)
    n_val = math.ceil(n_sample*ratio)  #上入整数
    n_train = n_sample - n_val

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    print("tra_images %d, tra_labels %d, val_images %d, val_labels %d"%(len(tra_images),len(tra_labels),
                                                                        len(val_images),len(val_labels)))

    return tra_images, tra_labels, val_images, val_labels

def make_namelist_lablelist(dir, doc, label):  #对文件夹中的图像做路径和标签列表
    '''
    参数：
    dir: 放不同类别的根目录
    doc: 子文件夹名称
    label:该文件夹对应的标签

    返回值:
    name：文件夹内的文件路径列表
    label_list_now: 标签列表
    '''
    name = []
    label_list_now = []
    doc_path = dir + '/' + doc + '/'
    sub_doc_list = os.listdir(doc_path)
    for sub_doc in sub_doc_list:
        name_now = doc_path + sub_doc
        name.append(name_now)
        label_list_now.append(label)

    return name, label_list_now

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    参数：
    image: list
    label: list
    image_W: 输入到网络的图像宽度
    imaga_H: 输入到网络的图像高度
    batch_size:
    capacity: queue的最大元素值
    返回值:
    imaga_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
    label_batch: 1D tensor [batch_size] dtype=tf.int32
    '''
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # 制作input queue
    input_queue = tf.train.slice_input_producer([image, label])

    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    label = input_queue[1]

    # 数据增强 #
    ###########

    # 将图像裁剪为制定大小 #
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 图像标准化 #
    image = tf.image.per_image_standardization(image)

    # capacity可以看成是局部数据的范围，读取的数据是基于这个范围的
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size = batch_size,
                                              num_threads = 64,
                                              capacity = capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch

'''
# 查看batch图像 #
import matplotlib.pyplot as plt

BATCH_SIZE = 2
CAPACITY = 256
IMG_W = 400
IMG_H = 400
trainval_dir = '/home/xinlong/Tensorflow_workspace/canjian_AlexNet/JPG/trainval'
RATIO = 0.2  # 验证集比例
tra_images, tra_labels, val_images, val_labels = get_files(trainval_dir, RATIO)
tra_image_batch, tra_label_batch = get_batch(tra_images, tra_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop() and i<2:
            img, label = sess.run([tra_image_batch, tra_label_batch])

            for j in range(BATCH_SIZE):
                print('label: %d' %label[j])
                plt.imshow(img[j,:,:,:])
                plt.show()
            i += 1
            print(i)

    except tf.error.OutofRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)

#get_files(trainval_dir, RATIO)

'''