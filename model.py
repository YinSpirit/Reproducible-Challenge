#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorflow.contrib import layers as l
import time
from ops import *

# from tensorflow.python.ops import variable_scope as vs
# from tensorflow.python.ops import math_ops, init_ops, array_ops, nn
# from tensorflow.python.util import nest
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell

# https://github.com/david-gpu/srez/blob/master/srez_model.py


def G_Depth(d_input,m, d_flg):
    
    use_bn = False
    base_filters = 32
    p = 0.5
    net = {}
    net['gate'] = d_input
    
    with tf.variable_scope('G_Depth'):

        net['conv1'] = conv2d(net['gate'], base_filters, 'convolution_1', kernel_sz=5, stride=1, act=tf.nn.elu, bn=use_bn)
        
        net['pool'] = l.avg_pool2d(net['conv1'], 3, 2, padding='SAME')

        net['conv2'] = projection_block(net['pool'], base_filters*2, 'convolution_2', kernel_sz=3,  act=tf.nn.elu, bn=use_bn)

        net['conv3'] = downscale_pool(net['conv2'], base_filters*2, 'convolution_3', kernel_sz=3,   act=tf.nn.elu, bn=use_bn)
        net['drop2'] = drop(net['conv3'],p, is_training=d_flg)

        #/4
        net['conv4'] = projection_block(net['drop2'], base_filters*4, 'convolution_4', kernel_sz=3, act=tf.nn.elu, bn=use_bn)
        net['conv5'] = downscale_pool(net['conv4'], base_filters*4, 'convolution_5', kernel_sz=3, act=tf.nn.elu, bn=use_bn)
        net['drop3'] = drop(net['conv5'],p, is_training=d_flg) 

        net['conv6'] = projection_block(net['drop3'], base_filters*8, 'convolution_6', kernel_sz=3, act=tf.nn.elu, bn=use_bn)
        net['conv7'] = residual_block(net['conv6'], base_filters*8, 'convolution_7', kernel_sz=3, act=tf.nn.elu, bn=use_bn)

        net['drop4'] = drop(net['conv7'],p, is_training=d_flg)

        net['convup1'] = conv2d(net['drop4'], base_filters*16, 'upconv_1',  stride=1, kernel_sz=3,act=tf.nn.elu, bn=False)
        net['uconv1'] = tf.depth_to_space(net['convup1'], 2)
        net['skip1'] =  tf.concat([net['conv4'], net['uconv1']], 3)

        net['convup2'] = conv2d(net['skip1'], base_filters*8, 'upconv_2',  stride=1, kernel_sz=3,act=tf.nn.elu, bn=False)
        net['uconv2'] = tf.depth_to_space(net['convup2'], 2)
        net['skip2'] =  tf.concat([net['conv2'], net['uconv2']], 3)  

        net['convup3'] = conv2d(net['skip2'], base_filters*4, 'upconv_3',  stride=1, kernel_sz=3,act=tf.nn.elu, bn=False)
        net['uconv3'] = tf.depth_to_space(net['convup3'], 2)
        net['skip3'] =  tf.concat([net['conv1'], net['uconv3']], 3)

        net['uconv4'] = conv2d(net['skip3'], base_filters, 'upconv_4',  stride=1, kernel_sz=3,act=tf.nn.elu, bn=False)

        net['residuals'] = net['uconv4'] 
        net['conv_out'] = conv2d(net['residuals'], 1, 'convolution_out', kernel_sz=3, act=None, bn=False)
    
    G_output = net['conv_out']  + net['gate'] + m

    print('>>>>>>>>>>>>>>>>>>>>>',G_output.shape)
    return G_output

def G_Depth_SR(d_input, t_interpolated, d_flg):

    base_filters = 32
    p = 0.5
    net = {}
    net['gate'] = d_input

    with tf.variable_scope('G_Depth_SR'):
        '''
        net['conv1'] = conv2d(net['gate'], base_filters, 'convolution_1', kernel_sz=3, stride=1, act=tf.nn.elu)
        net['conv2'] = projection_block(net['conv1'], base_filters*2, 'convolution_2', kernel_sz=3, act=tf.nn.elu)
        '''
        net['conv1'] = residual_block(net['gate'],  base_filters*2, 'convolution_1', kernel_sz=3, act=tf.nn.elu, bn = False)
        net['conv2'] = residual_block(net['conv1'], base_filters*2, 'convolution_2', kernel_sz=3, act=tf.nn.elu, bn = False)
        net['drop1'] = drop(net['conv2'], p, is_training=d_flg)
        net['conv3'] = residual_block(net['drop1'], base_filters*2, 'convolution_3', kernel_sz=3, act=tf.nn.elu, bn = False)
        net['conv4'] = residual_block(net['conv3'], base_filters*2, 'convolution_4', kernel_sz=3, act=tf.nn.elu, bn = False)
        net['drop2'] = drop(net['conv4'], p, is_training=d_flg)
        net['conv5'] = residual_block(net['drop2'], base_filters*2, 'convolution_5', kernel_sz=3, act=tf.nn.elu, bn = False)
        net['conv6'] = residual_block(net['conv5'], base_filters*2, 'convolution_6', kernel_sz=3, act=tf.nn.elu, bn = False)
        net['drop3'] = drop(net['conv6'], p, is_training=d_flg)
        net['conv7'] = conv2d(net['drop3'], base_filters*4, 'upconv_1',  stride=1, kernel_sz=3,act=tf.nn.elu, bn = False)
        net['uconv1'] = tf.depth_to_space(net['conv7'], 2)
        net['conv8'] = conv2d(net['uconv1'], base_filters*2,'upconv_2',  stride=1, kernel_sz=3,act=tf.nn.elu, bn = False)
        net['uconv2'] = tf.depth_to_space(net['conv8'], 2)
        net['conv9'] = deconv2d(net['uconv2'], base_filters,'upconv_3',  stride=1, kernel_sz=3,act=tf.nn.elu, bn = False)
        net['uconv3'] = tf.depth_to_space(net['conv9'], 2)
        net['conv_out'] = conv2d(net['uconv3'], 1, 'convolution_out', kernel_sz=3, act=None, bn=False)
    
    G_output = net['conv_out'] + t_interpolated

    print('>>>>>>>>>>>>>>>>>>>>>',G_output.shape)
    return G_output

def Vgg19_simple_api(rgb, reuse):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else:  # TF 1.0
            # print(rgb_scaled)
            red, green, blue = tf.split(rgb_scaled, 3, 3)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat(
                [
                    blue - VGG_MEAN[0],
                    green - VGG_MEAN[1],
                    red - VGG_MEAN[2],
                ], axis=3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        """ input layer """
        net_in = tl.layers.InputLayer(bgr, name='input')
        """ conv1 """
        network = tl.layers.Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
        network = tl.layers.Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
        network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        """ conv2 """
        network = tl.layers.Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
        network = tl.layers.Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
        network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
        """ conv3 """
        network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
        network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
        network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
        network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
        network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
        """ conv4 """
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')
        network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')  # (batch_size, 14, 14, 512)
        conv = network
        """ conv5 """
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')
        network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')  # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        network = tl.layers.FlattenLayer(network, name='flatten')
        network = tl.layers.DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
        network = tl.layers.DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
        network = tl.layers.DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        print("build model finished: %fs" % (time.time() - start_time))
        return network, conv

