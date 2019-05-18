import tensorlayer as tl
import tensorflow as tf

from model import G_Depth_SR
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

from skimage.measure import compare_ssim

import warnings
warnings.filterwarnings('ignore')

###====================== CONFIG-FILES ===============================###
small_size = [38, 64]
big_size = [304, 512]
# Path to Train Datasets
checkpoint_dir1 = 'Precomputed_weights/depth sr'
mid_dir = 'Datasets/Middlebury/'
art = mid_dir+'art/disp1.png'
laundry = mid_dir+'laundry/disp1.png'
moebius = mid_dir+'moebius/disp1.png'
dolls = mid_dir+'dolls/disp1.png'

# def load_png(png):
#     d = cv2.imread(png, flags=cv2.IMREAD_ANYDEPTH).astype('float32')

def preprocess_SR(d_tensor, size_small, size_big):
    
    d_log = (d_tensor - tf.reduce_min(d_tensor)) / (tf.reduce_max(d_tensor)- 
                                                    tf.reduce_min(d_tensor))
    d_log = d_log + tf.constant(1.)
    d_small = tf.image.resize_images(d_log, size_small, method=tf.image.ResizeMethod.AREA)
    d_upsampled = tf.image.resize_images(d_small, size_big, method=tf.image.ResizeMethod.BILINEAR)
    #ground_truth = d_log
    ground_truth = tf.image.resize_images(d_log, size_big, method=tf.image.ResizeMethod.BILINEAR)
    
    return ground_truth, d_small, d_upsampled

def rmse_f(x, y):
    return np.sqrt(((x - y)*(x - y)).mean())

def norm_max(img):
    return (img - img.min())/(img.max() - img.min())

def open_middlebury_disp(fname):
    d = cv2.imread(fname,cv2.IMREAD_ANYDEPTH)
    mask = np.isinf(d)
    min_d = d.min()
    max_d = d[np.logical_not(mask)].max()
    d = (d - min_d) / (max_d - min_d) * 255
    d[mask] = np.zeros_like(d)[mask]
    idx = np.where(d == 0)
    d_b = cv2.medianBlur(d.astype(np.uint8), 155)
    d[idx] = d_b[idx]
    d = cv2.resize(d, (2780, 1920))
    d = d.astype(np.float32) * (max_d - min_d) + min_d
    return d


def evaluate_exp1():

    target = tf.placeholder(tf.float32, shape=(None,None,None, 1))
    d_flg = tf.placeholder(tf.bool)

    y_true, d_input, interpolated = preprocess_SR(target, small_size, big_size)

    G_output = G_Depth_SR(d_input, interpolated, d_flg)
    
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir1))

    middle2 = [art, laundry, moebius, dolls]
    rmse_results = []
    ssim_results = []
    for p in middle2:

        d = cv2.resize(cv2.imread(p, flags=cv2.IMREAD_ANYDEPTH).astype('float32'), tuple(big_size))
    
        time_start = time.time()

        d_big, d_sm, interp, out = sess.run([y_true, d_input, interpolated, G_output], 
                                       feed_dict={target:d[None,:,:,None], d_flg:False})
        print(f'Time: {time.time() - time_start}')
    
        out_norm = norm_max(out[0,:,:,0])
        d_norm = norm_max(d_big[0,:,:,0])
    
        print(d_norm.shape, out_norm.shape)
    
        temp1 = (out_norm * 255).astype('uint8')
        temp2 = (d_norm * 255).astype('uint8')
    
        print(p[:-10])
        print('rmse = ', round(rmse_f(temp1, temp2),2))
        print('ssim=', round(compare_ssim(d_norm, out_norm, data_range= 1),2))
        rmse_results.append(round(rmse_f(temp1, temp2),2))
        ssim_results.append(round(compare_ssim(d_norm, out_norm, data_range= 1),2))

        if (p == art):
            fig, ax = plt.subplots(nrows=1, ncols=3)
            fig.set_size_inches(15,5)
            plt.setp(ax, xticks=[], yticks=[])
            ax[0].matshow(d_big[0,:,:,0],cmap ='magma'), ax[0].set_title("ground_truth"),ax[0].axis('off') 
            ax[1].matshow(d_sm[0,:,:,0],cmap ='magma'), ax[1].set_title("input"), ax[1].axis('off') 
            ax[2].matshow(out_norm,cmap ='magma'), ax[2].set_title("output"), ax[2].axis('off') 

            fig.savefig('Results/SR/sr_results.png', bbox_inches='tight')
    
    row_data3 = r'''
    \begin{table*}
    \centering
    \label{sr}
    \begin{tabular}{l|c|c|c|c|c|c|c|c|}
    \cline{2-9}
    \multicolumn{1}{c|}{} & \multicolumn{2}{c|}{Art} & \multicolumn{2}{c|}{Laundry} & \multicolumn{2}{c|}{Moebius} & \multicolumn{2}{c|}{Dolls} \\ \cline{2-9} 
    \multicolumn{1}{c|}{} & SSIM & RMSE & SSIM & RMSE & SSIM & RMSE & SSIM & RMSE \\ \hline
    \multicolumn{1}{|l|}{Yang \cite{yangdict}} & 0.43 & 46.21 & 0.41 & 33.87 & 0.41 & 19.01 & 0.54 & 16.42 \\ \hline
    \multicolumn{1}{|l|}{Wang \cite{wangups}} & 0.55 & 47.07 & 0.53 & 32.56 & 0.54 & 19.76 & 0.63 & 15.16 \\ \hline
    \multicolumn{1}{|l|}{Konno et al. \cite{self-guided}} & 0.63 & 30.01 & 0.59 &  21.31 & 0.61 & 12.09 & 0.70 & 10.86 \\ \hline
    \multicolumn{1}{|l|}{Our Method} & %(var00)s & %(var01)s & %(var02)s & %(var03)s & %(var04)s & %(var05)s & %(var06)s & %(var07)s \\ \hline
    \end{tabular}
    \caption{Evaluation of performance of the super-resolution network. Results are given for $\times8$ upsampling; for other methods values obtained from \cite{self-guided}}
    \end{table*}
    '''

    var_dict = {'var00': rmse_results[0], 'var01': ssim_results[0], 'var02': rmse_results[1], 'var03': ssim_results[1], 
                'var04': rmse_results[2], 'var05': ssim_results[2], 'var06': rmse_results[3], 'var07': ssim_results[3]}
    

    b = row_data3 % var_dict
    f = open('Results/SR/table_sr_exp1.tex', 'w')
    f.write(b)
    f.close()
    print(b)

    sess.close()




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    exp_list = ['exp1']

    parser.add_argument('--mode', type=str, default='exp1', help='SR')
    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'exp1':
        print("Running experiment 1")
        evaluate_exp1()
    else:
        raise Exception("Unknown --mode")
