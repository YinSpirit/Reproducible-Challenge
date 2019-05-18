#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, random
from datetime import datetime
import numpy as np
import cv2
from iterators import *

import tensorflow as tf
import tensorlayer as tl
from model import G_Depth, G_Depth_SR, Vgg19_simple_api
from utils import *
from skimage.filters import sobel_h, sobel_v
from skimage.color import rgb2grey
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from tqdm import tqdm
from vgg.vgg16 import VGG16
import random

###====================== CONFIG-FILES ===============================###
train_synthia_size = [152, 256]
small_synthia_size = [38,64]
big_synthia_size = [304,512]

train_int_batch_size = 8
train_sr_batch_size = 5
train_rate = [1e-4]*30 + [1e-5]*30 
train_eps = 1e-2
train_maxi = 1e4
train_int_n_epoch = 35
train_sr_n_epoch = 12
# Path to Train Datasets
train_hr_img_path = 'Datasets/synthia/train'
val_hr_img_path = 'Datasets/synthia/SYNTHIA-SEQS-05-SPRING'

###====================== HYPER-PARAMETERS ===========================###
batch_int_size = train_int_batch_size
batch_sr_size = train_sr_batch_size
eps = tf.constant(train_eps)
rate = train_rate
epsi = 1e-6
maxi = tf.constant(train_maxi)
n_int_epoch = train_int_n_epoch
n_sr_epoch = train_sr_n_epoch

synthia_size = train_synthia_size
vgg16_npy_path = "vgg/vgg16.npy"



def combined_mask(img, sz, combined=True):
    resized = cv2.resize(img.astype(np.uint8), tuple(sz[::-1]))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0.2)
    grey = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(grey, cv2.CV_16S, 1, 0).astype(float)
    grad_y = cv2.Sobel(grey, cv2.CV_16S, 0, 1).astype(float)
    grad = np.sqrt(np.square(grad_x) + np.square(grad_y))
    grad = np.round((grad - grad.min())/(grad.max() - grad.min())*255).astype(np.uint8)
    grad_thr = cv2.adaptiveThreshold(grad,np.max(grad), cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 11)
    grad_thr = (grad_thr - grad_thr.min())/(grad_thr.max() - grad_thr.min())
    grad_thr = np.ones(grad_thr.shape, dtype=np.float32) - grad_thr.astype(np.float32) 
    random = np.random.choice([0, 1],size=sz, p=[1 - 0.05, 0.05])
    if combined:
        grad_thr = np.clip(grad_thr + random, 0, 1)
    else:
        grad_thr = np.clip(grad_thr, 0, 1)
    return grad_thr.astype(int)[None, :, :, None]

def get_grad_mask(img, sz, t=51, combined=True):
    bs = img.shape[0]
    return np.concatenate([combined_mask(img[i,:,:,:], sz, combined=combined) for i in range(bs)])

def get_mask(sz, p):

    mask = np.random.choice([0,1],size=sz, p=[1 - p, p])[None,:,:,None]
    return mask

def get_regular_grid_mask(sz, step = 7):
    z = np.zeros((1, *sz, 1))
    z[:,::step,::step,:] = 1
    return z

def preprocess_SR(d_tensor, size_small, size_big):
    
    d_log = (d_tensor - tf.reduce_min(d_tensor)) / (tf.reduce_max(d_tensor)- 
                                                    tf.reduce_min(d_tensor))
    d_log = d_log + tf.constant(1.)
    d_small = tf.image.resize_images(d_log, size_small, method=tf.image.ResizeMethod.AREA)
    d_upsampled = tf.image.resize_images(d_small, size_big, method=tf.image.ResizeMethod.BILINEAR)
    #ground_truth = d_log
    ground_truth = tf.image.resize_images(d_log, size_big, method=tf.image.ResizeMethod.BILINEAR)
    
    return ground_truth, d_small, d_upsampled

def preprocess(rgb_tensor, d_tensor, mask_tensor,size, eps, maxi):

    
    rgb_res = tf.image.resize_images(rgb_tensor, size)
    greyscale =  rgb_res[:,:,:,0] * 0.2989 + rgb_res[:,:,:,1] * 0.5870 + rgb_res[:,:,:,2] * 0.1140 
    greyscale = greyscale[:,:,:, None]
    g_min= tf.reduce_min(greyscale, axis=(1,2), keep_dims=True)
    g_max= tf.reduce_max(greyscale, axis=(1,2), keep_dims=True)
    g_normed = (greyscale - g_min)/(g_max - g_min)
    g_out = g_normed - tf.reduce_mean(g_normed, axis=(1,2), keep_dims=True)
    d_cl = tf.clip_by_value(d_tensor, eps, maxi)
    d_log = tf.log(d_cl + 1.)
    d_res = tf.image.resize_images(d_log, size)
    d_norm = d_res
    nonzero = tf.cast(tf.count_nonzero(d_norm*mask_tensor[:,:], axis = (1,2), keep_dims= True), tf.float32)
    
    masked_sum = tf.reduce_sum(d_norm*mask_tensor[:,:], axis = (1,2), keep_dims= True)
    mean_d = masked_sum/nonzero
    ground_truth = d_norm
    
    mean, std = tf.nn.moments(ground_truth*mask_tensor[:,:], axes=(1,2), keep_dims=True)
    d_masked_out = (ground_truth*mask_tensor[:,:] - mean_d)
    
    return ground_truth, g_out, d_masked_out, mean_d

def train_int(mask_str, loss_str):
    checkpoint_dir = "checkpoint/g_depth_int"  # checkpoint_resize_conv
    log_dir = "logs/g_depth_int"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    print("At pre load stage")
    synthia_depth, synthia_rgb = get_synthia_imgs_list(val_hr_img_path, is_train=True, synthia_dataset=train_hr_img_path, is_rgb = True)
    synthia_depth_val, synthia_rgb_val = get_synthia_imgs_list(val_hr_img_path, is_train=False, synthia_dataset=train_hr_img_path, is_rgb = True)
    print(len(synthia_depth), len(synthia_rgb))
    print(len(synthia_depth_val), len(synthia_rgb_val))

    #initializing some of photos from val set
    val_image_s = open_rgb(synthia_rgb_val[0])[None,:,:,:]
    val_gt_s = open_depth_synthia(synthia_depth_val[0], debug=True)[None]
    val_image_d = open_rgb(synthia_rgb_val[-1])[None,:,:,:]
    val_gt_d = open_depth_synthia(synthia_depth_val[-1])[None,:,:,:]
    #getting grad masks
    grad_ms = get_grad_mask(val_image_s,synthia_size)
    grad_md = get_grad_mask(val_image_d,synthia_size)
    print("Passed loading stage")


    ###========================== DEFINE MODEL ============================###
    ## train inference
    input_rgb = tf.placeholder(tf.float32, shape=(None,None,None, 3))
    target = tf.placeholder(tf.float32, shape=(None,None,None, 1))
    mask_t = tf.placeholder(tf.float32, shape=(None,None,None,1))
    d_flg = tf.placeholder(tf.bool)

    y_true, gr_input, d_input, m = preprocess(input_rgb, target, mask_t, synthia_size, eps, maxi)

    G_output = G_Depth(d_input, m, d_flg)

    vgg_model_true = VGG16(vgg16_npy_path)
    vgg_model_gen = VGG16(vgg16_npy_path)

    with tf.variable_scope('loss'):

        exp_out = G_output 
        exp_true = y_true
        y_true_normalized = (exp_true - tf.reduce_min(exp_true))/(tf.reduce_max(exp_true) 
                                                              - tf.reduce_min(exp_true))
        y_true_3ch = tf.concat([y_true_normalized]*3,3)
    
        gen_normalized = (exp_out - tf.reduce_min(exp_out))/(tf.reduce_max(exp_out) 
                                                         - tf.reduce_min(exp_out))
        gen_normalized = (exp_out - tf.reduce_min(exp_out))/(tf.reduce_max(exp_out) 
                                                         - tf.reduce_min(exp_out))
        generated_3ch = tf.concat([gen_normalized]*3, 3)
    
        vgg_model_true.build(y_true_3ch)
        true_features = vgg_model_true.conv3_1
        vgg_model_gen.build(generated_3ch)
        gen_features = vgg_model_gen.conv3_1
        VGG_loss = tf.reduce_mean(tf.square(true_features - gen_features))

        #generator_loss
        tv_loss = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.square(G_output[:, :-1, -1:, :] - G_output[:, 1:, -1:, :]) + \
                tf.square(G_output[:, :-1, :-1, :] - G_output[:, :-1, 1:, :])), axis=[1, 2, 3]))
        square_loss = tf.reduce_mean(tf.square(y_true - G_output))
        if (loss_str == 'MSE_vgg1'):
            G_loss = square_loss + 1e-2*VGG_loss
        elif (loss_str == 'MSE_vgg2'):
            G_loss = square_loss + 5e-5*VGG_loss
        elif (loss_str == 'MSE_only'):
            G_loss = square_loss
        else:
            G_loss = square_loss + 5e-5*VGG_loss + 1e-5*tv_loss

    with tf.variable_scope('opt'):
        #optimizers
        all_params = tf.trainable_variables()
        first_stage = [p for p in all_params if 'G_Depth' in p.name]
        lr = tf.placeholder(dtype=tf.float32)
        train_G = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list = first_stage)

    saver = tf.train.Saver(max_to_keep=1)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)

    with tf.variable_scope('summaries'):
        loss = tf.summary.scalar('loss', G_loss)
        mae = tf.reduce_mean(tf.abs(G_output-y_true)/(y_true + eps))
        rmse = tf.sqrt(tf.reduce_mean(tf.square(G_output-y_true)))
        mae_summary = tf.summary.scalar('MAE', mae)
        in_summary = tf.summary.image('input', d_input , max_outputs=1)
        out_summary = tf.summary.image('output',G_output , max_outputs=1)
        gt_summary = tf.summary.image('gt_depth', y_true , max_outputs=1)
        all_summaries = tf.summary.merge([mae_summary, in_summary, out_summary, loss])

    sess = tf.InteractiveSession()
    ground_truths = tf.summary.FileWriter(log_dir + '/train' + '/img_gt_s')
    ground_truthd = tf.summary.FileWriter(log_dir + '/train'+ '/img_gt_d')
    writer_grads = tf.summary.FileWriter(log_dir + '/train'+ '/grad_s')
    writer_gradd = tf.summary.FileWriter(log_dir + '/train'+ '/grad_d')
    sess.run(tf.global_variables_initializer())
    ###============================= TRAINING ===============================###

    n_batches = int(len(synthia_depth) / batch_int_size)
    n_batches_valid = int(len(synthia_depth_val) / batch_int_size)

    ###========================= initialize G_depth ====================###
    ground_truths.add_summary(gt_summary.eval({input_rgb:val_image_s, target:val_gt_s}))
    ground_truthd.add_summary(gt_summary.eval({input_rgb:val_image_d, target:val_gt_d}))
    i = 0
    #first we will have test initializing
    for e in range(n_int_epoch):
        try:
            epoch_time = time.time()
            val_loss = 0
            batch_it = tqdm(SynthiaIterator(synthia_rgb_val, synthia_depth_val, batchsize=batch_int_size, shuffle=True, buffer_size=70),
                        total=n_batches_valid, leave=False)

            for xb, yb in batch_it:
                if (mask_str == 'regular_grid'):
                    mask = get_regular_grid_mask(synthia_size)  
                elif (mask_str == 'uniform'):
                    p = np.random.choice([0.05, 0.1, 0.2, 0.4])
                    mask = get_mask(synthia_size, p)
                elif (mask_str == 'gradient'):
                    mask = get_grad_mask(xb, synthia_size, combined = False)
                else:
                    mask = get_grad_mask(xb, synthia_size, combined = True)
                #along gradient + uniform
                # mask = get_grad_mask(xb, synthia_size, combined = True)
                #along only gradient
                # mask = get_grad_mask(xb, synthia_size, combined = False)
                #regular grid
                #mask = get_regular_grid_mask(synthia_size)  
                #just uniform
                #p = np.random.choice([0.05, 0.1, 0.2, 0.4])
                #mask = get_mask(synthia_size, p)

                ll = sess.run(mae, feed_dict={input_rgb:xb, target:yb, mask_t:mask, d_flg:False, 
                                                           lr:0})
                val_loss += ll
            print("epoch {0} val loss {1}".format(e - 1, val_loss/n_batches_valid))
        
            mse_loss = 0
            batch_it = tqdm(SynthiaIterator(synthia_rgb, synthia_depth, 
                                            batchsize=batch_int_size, 
                                            shuffle=True, 
                                            buffer_size = 70), 
                                            total=n_batches, 
                                            leave=False)
            for xb, yb in batch_it:
                if (mask_str == 'regular_grid'):
                    mask = get_regular_grid_mask(synthia_size)  
                elif (mask_str == 'uniform'):
                    p = np.random.choice([0.05, 0.1, 0.2, 0.4])
                    mask = get_mask(synthia_size, p)
                elif (mask_str == 'gradient'):
                    mask = get_grad_mask(xb, synthia_size, combined = False)
                else:
                    mask = get_grad_mask(xb, synthia_size, combined = True)
                #along gradient + uniform
                # mask = get_grad_mask(xb, synthia_size, combined = True)
                #along only gradient
                # mask = get_grad_mask(xb, synthia_size, combined = False)
                #regular grid
                #mask = get_regular_grid_mask(synthia_size)
                #just uniform
                #p = np.random.choice([0.05, 0.1, 0.2, 0.4])
                #mask = get_mask(synthia_size, p)
                aug = np.random.choice([0, 1, 2, 3])
                if aug == 1:
                    xb = xb[:,:,::-1,:]
                    yb = yb[:,:,::-1,:]
                    mask = mask[:,:,::-1,:]
                if aug == 2:
                    xb = xb[:,::-1,:,:]
                    yb = yb[:,::-1,:,:]
                    mask = mask[:,::-1,:,:]
                if aug == 3:
                    xb = xb[:,::-1,::-1,:]
                    yb = yb[:,::-1,::-1,:]
                    mask = mask[:,::-1,::-1,:]
                _, ll = sess.run([train_G, G_loss], feed_dict={input_rgb:xb, target:yb, mask_t:mask, d_flg:True, lr:rate[e]})
                mse_loss += ll
                if (i + 1)%10 == 0:
                    sum_grad = all_summaries.eval({input_rgb:val_image_s, target:val_gt_s, mask_t:grad_ms, d_flg:False})
                    writer_grads.add_summary(sum_grad, i)
                    sum_grad = all_summaries.eval({input_rgb:val_image_d, target:val_gt_d, mask_t:grad_md, d_flg:False})
                    writer_gradd.add_summary(sum_grad, i)
                i += 1

            saver.save(sess,f'{checkpoint_dir}/{mask_str}/{loss_str}/model{str(e)}.ckpt', global_step=e)
            print("epoch {0} train loss {1}, time {2}".format(e, mse_loss/n_batches, time.time() - epoch_time))    

        except Exception as e:
            batch_it.iterable.stop()
            raise e
            break 
    # save model after training
    sess.close()

def train_sr():
    checkpoint_dir = "checkpoint"  
    log_dir = "logs/g_depth_sr"
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    print("At pre load stage")
    synthia_depth = get_synthia_imgs_list(val_hr_img_path, is_train=True, synthia_dataset=train_hr_img_path, is_rgb = False)
    synthia_depth_val = get_synthia_imgs_list(val_hr_img_path, is_train=False, synthia_dataset=train_hr_img_path, is_rgb = False)
    print(len(synthia_depth), len(synthia_depth_val))

    val_gt_s = open_depth_synthia(synthia_depth_val[0], debug=True)[None]
    val_gt_d = open_depth_synthia(synthia_depth_val[-1])[None,:,:,:]
    print("Passed loading stage")


    ###========================== DEFINE MODEL ============================###
    target = tf.placeholder(tf.float32, shape=(None,None,None, 1))
    d_flg = tf.placeholder(tf.bool)

    y_true, d_input, interpolated = preprocess_SR(target, small_synthia_size, big_synthia_size)

    G_output = G_Depth_SR(d_input, interpolated, d_flg)

    vgg_model_true = VGG16(vgg16_npy_path)
    vgg_model_gen = VGG16(vgg16_npy_path)

    with tf.variable_scope('loss'):

        exp_out = G_output 
        exp_true = y_true
        y_true_normalized = (exp_true - tf.reduce_min(exp_true))/(tf.reduce_max(exp_true) 
                                                              - tf.reduce_min(exp_true))
        y_true_3ch = tf.concat([y_true_normalized]*3,3)
    
        gen_normalized = (exp_out - tf.reduce_min(exp_out))/(tf.reduce_max(exp_out) 
                                                         - tf.reduce_min(exp_out))
        gen_normalized = (exp_out - tf.reduce_min(exp_out))/(tf.reduce_max(exp_out) 
                                                         - tf.reduce_min(exp_out))
        generated_3ch = tf.concat([gen_normalized]*3, 3)
    
        vgg_model_true.build(y_true_3ch)
        true_features = vgg_model_true.conv3_1
        vgg_model_gen.build(generated_3ch)
        gen_features = vgg_model_gen.conv3_1
        VGG_loss = tf.reduce_mean(tf.square(true_features - gen_features))

        #loss
        tv_loss = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.square(G_output[:, :-1, -1:, :] - G_output[:, 1:, -1:, :]) + \
                tf.square(G_output[:, :-1, :-1, :] - G_output[:, :-1, 1:, :])), axis=[1, 2, 3]))
        square_loss = tf.reduce_mean(tf.square(y_true - G_output))
        G_loss = square_loss + 5e-5*VGG_loss + 2e-6*tv_loss

    with tf.variable_scope('opt'):
        #optimizers
        all_params = tf.trainable_variables()
        first_stage = [p for p in all_params if 'G_Depth_SR' in p.name]
        lr = tf.placeholder(dtype=tf.float32)
        train_G = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list = first_stage)

    saver = tf.train.Saver(max_to_keep=1)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)

    with tf.variable_scope('summaries'):
        loss = tf.summary.scalar('loss', G_loss)
        mae = tf.reduce_mean(tf.abs(G_output-y_true)/(y_true + eps))
        rmse = tf.sqrt(tf.reduce_mean(tf.square(G_output-y_true)))
        mae_summary = tf.summary.scalar('MAE', mae)
        in_summary = tf.summary.image('input', d_input , max_outputs=1)
        out_summary = tf.summary.image('output',G_output , max_outputs=1)
        gt_summary = tf.summary.image('gt_depth', y_true , max_outputs=1)
        all_summaries = tf.summary.merge([mae_summary, in_summary, out_summary, loss])

    sess = tf.InteractiveSession()
    ground_truths = tf.summary.FileWriter(log_dir + '/train' + '/img_gt_s')
    ground_truthd = tf.summary.FileWriter(log_dir + '/train'+ '/img_gt_d')
    writer_grads = tf.summary.FileWriter(log_dir + '/train'+ '/grad_s')
    writer_gradd = tf.summary.FileWriter(log_dir + '/train'+ '/grad_d')
    sess.run(tf.global_variables_initializer())

    n_batches = int(len(synthia_depth) / batch_sr_size)
    n_batches_valid = int(len(synthia_depth_val) / batch_sr_size)

    ###========================= initialize G_depth_SR ====================###
    ground_truths.add_summary(gt_summary.eval({target:val_gt_s}))
    ground_truthd.add_summary(gt_summary.eval({target:val_gt_d}))
    i = 0
    #first we will have test initializing
    for e in range(1):
        try:
            epoch_time = time.time()
            val_loss = 0
            batch_it = tqdm(SynthiaIterator(synthia_depth_val, batchsize=batch_sr_size, shuffle=True, buffer_size=70),
                        total=n_batches_valid, leave=False)

            for b in batch_it:
                xb = b[0]
                aug = np.random.choice([0, 1])
                if aug == 1:
                    xb = xb[:,:,::-1,:]

                ll = sess.run(mae, feed_dict={target:xb, d_flg:False,lr:0})
                val_loss += ll

            print("epoch {0} val loss {1}".format(e - 1, val_loss/n_batches_valid))
        
            mse_loss = 0
            batch_it = tqdm(SynthiaIterator(synthia_depth, 
                                            batchsize=batch_sr_size, 
                                            shuffle=True, 
                                            buffer_size = 70), 
                                            total=n_batches, 
                                            leave=False)
            for b in batch_it:
                xb = b[0]
                aug = np.random.choice([0, 1])
                if aug == 1:
                    xb = xb[:,:,::-1,:]
                _, ll = sess.run([train_G, G_loss], feed_dict={target:xb, d_flg:True, lr:rate[e]})
                mse_loss += ll
                if (i + 1)%10 == 0:
                    sum_grad = all_summaries.eval({target:val_gt_s, d_flg:False})
                    writer_grads.add_summary(sum_grad, i)
                    sum_grad = all_summaries.eval({target:val_gt_d, d_flg:False})
                    writer_gradd.add_summary(sum_grad, i)
                i += 1

            saver.save(sess,f'{checkpoint_dir}/g_depth_sr/model{str(e)}.ckpt', global_step=e)
            print("epoch {0} train loss {1}, time {2}".format(e, mse_loss/n_batches, time.time() - epoch_time))    

        except Exception as e:
            batch_it.iterable.stop()
            raise e
            break 
    # save model after training
    sess.close()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    mask_list = ['combined','gradient', 'uniform', 'regular_grid']
    loss_list = ['MSE_vgg1','MSE_vgg2', 'MSE_only', 'MSE_vggTV']

    parser.add_argument('--mode', type=str, default='gdepth_int', help='gdepth')
    parser.add_argument('--mask', type=str, default='combined', help='mask_h')
    parser.add_argument('--loss', type=str, default='MSE_vggTV', help='loss_h')
    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    tl.global_flag['mask'] = args.mask
    tl.global_flag['loss'] = args.loss

    if tl.global_flag['mask'] not in mask_list:
        raise Exception("Unknown --mask")

    if tl.global_flag['loss'] not in loss_list:
        raise Exception("Unknown --loss")

    if tl.global_flag['mode'] == 'gdepth_int':
        print("Training Depth Interpolation NN")
        train_int(tl.global_flag['mask'], tl.global_flag['loss'])
    elif tl.global_flag['mode'] == 'gdepth_sr':
        print("Training Depth Super Resolution NN")
        train_sr()
        #pass
    else:
        raise Exception("Unknow --mode")
