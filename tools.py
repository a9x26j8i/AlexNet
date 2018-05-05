# -*- coding:utf-8 -*-
### 卷积层 池化层 loss 等函数###
import tensorflow as tf
import numpy as np
import os

def conv_layer(layer_name, input_img, filter_size, strides, in_channels, out_channels, activation = True):
    '''
    参数：
    layer_name: 卷积层名称
    input_img: 输入特征
    filter_size: 卷积核大小
    strides:     卷积步长
    in_channels: 输入特征图通道数
    out_channels: 输出特征图通道数
    activation: 激活函数 （1 为ReLU， 0为没有激活函数）

    返回值:
    卷积后的特征图
    '''
    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[filter_size, filter_size, in_channels, out_channels],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32))
        b = tf.get_variable('bias',
                            shape=[out_channels],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_img, w, strides=[1,strides,strides,1], padding='SAME',name='conv')
        pre_activation = tf.nn.bias_add(conv, b, name='conv_add_bias')

        if activation == True:
            conv_out = tf.nn.relu(pre_activation, name='relu')
        else:
            conv_out = pre_activation
        return conv_out

def pool(p_layer_name, p_input, p_ksize, p_strides, is_max_pool=True):
    '''
    参数
    p_layer_name: 池化层名称
    p_input:      输入特征图
    p_ksize:      池化大小
    p_strides:    池化步长
    is_max_pool:  是否使用最大池化方式

    返回值:        池化后的特征
    '''

    with tf.variable_scope(p_layer_name):
        if is_max_pool:
            pool_out = tf.nn.max_pool(p_input,ksize=[1,p_ksize,p_ksize,1],
                                  strides=[1,p_strides,p_strides,1],
                                  padding='SAME',name=p_layer_name)
        else:
            pool_out = tf.nn.avg_pool(p_input, ksize=[1, p_ksize, p_ksize, 1],
                                  strides=[1, p_strides, p_strides, 1],
                                  padding='SAME', name=p_layer_name)
    return pool_out

def FC_layer(layer_name, fc_input, out_nodes, if_relu=True):
    '''
    参数：
    layer_name: 全连接层名称
    fc_input:   输入
    out_nodes:  输出节点数

    返回值:      全连接输出
    '''
    shape = fc_input.get_shape()
    # shape = [batchsize, H, W, channels]
    if len(shape) == 4:
        size = shape[1] * shape[2] * shape[3]
    else:
        size = shape[-1].value # ???
    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('bias',
                            shape= [out_nodes],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(fc_input, [-1, int(size)])
        multiply = tf.matmul(flat_x, w)
        if if_relu == True:
            fc_out = tf.nn.relu(tf.nn.bias_add(multiply, b))
        else:
            fc_out = tf.nn.bias_add(multiply, b)

        return fc_out

def loss(loss_name, labels, logits):
    '''
    参数：
    loss_name:  损失层名字
    labels:     标签数据   [batchsize]
    logits:     预测数据   [batchsize, n_classes]

    返回值:      loss值
    '''

    with tf.variable_scope(loss_name) as scope:
        '''
        shape_logits = logits.get_shape()
        shape_labels = labels.get_shape()
        print('shape logits', shape_logits)
        print('shape labels', shape_labels)
        '''
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar('loss_', loss)
        return loss

def accuracy(acc_name, labels, logits):
    '''
    参数：
    acc_name:  准确率层名称
    labels:    标签值  [batchsize]
    logits:    判断值  [batchsize, num_class]]

    返回值:     准确率
    '''
    with tf.variable_scope(acc_name) as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy/', accuracy)
    return accuracy

def optimize(opt_name, loss, learning_rate, global_step):
    '''
    参数：
    opt_name:      优化层名称
    loss:          损失函数
    learning_rate: 学习率
    global_step:   迭代次数
    
    返回值:         训练op
    '''
    with tf.variable_scope(opt_name):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
