# -*- coding:utf-8 -*-
import tensorflow as tf
import tools

def AlexNet(x, n_class):


    x = tools.conv_layer('conv1', x, filter_size=11, strides=4, in_channels=3, out_channels=96, activation=True)

    x = tools.pool('pool1', x, p_ksize=3, p_strides=2, is_max_pool=True)

    x = tools.conv_layer('conv2', x, filter_size=5, strides=1, in_channels=96, out_channels=256, activation=True)

    x = tools.pool('pool2', x, p_ksize=3, p_strides=2, is_max_pool=True)

    x = tools.conv_layer('conv3', x, filter_size=3, strides=1, in_channels=256, out_channels=384, activation=True)

    x = tools.conv_layer('conv4', x, filter_size=3, strides=1, in_channels=384, out_channels=384, activation=True)

    x = tools.conv_layer('conv5', x, filter_size=3, strides=1, in_channels=384, out_channels=256, activation=True)

    x = tools.pool('pool3', x, p_ksize=3, p_strides=2, is_max_pool=True)

    x = tools.FC_layer('FC1', x, out_nodes=4096, if_relu=True)
    x = tools.FC_layer('FC2', x, out_nodes=4096, if_relu=True)

    x = tools.FC_layer('FC3', x, out_nodes=n_class, if_relu=None)
    return x

''''
# 测试输出的维度是否正确
import matplotlib.pyplot as plt
import tools
import matplotlib
matplotlib.use('Agg')

#%%
cat = plt.imread('cat.jpg') #unit8
#plt.imshow(cat)
cat = tf.cast(cat, tf.float32) #[360, 300, 3]
x = tf.reshape(cat, [1, 650, 948, 3]) #[1, 360, 300, 3]

def shape(x):
    return (x.get_shape())

logits = AlexNet(x, 5)

print(shape(logits))
'''