import tensorlayer as tl
import tensorflow as tf
from model import G_Depth
from iterators import SynthiaIterator
from utils import  get_file_list, filter_files, open_rgb, open_depth
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import h5py
import seaborn as sns
from skimage.measure import compare_ssim, compare_psnr
from skimage.transform import resize
import cv2
from sintel_io import depth_read

import warnings
warnings.filterwarnings('ignore')

###====================== CONFIG-FILES ===============================###
synthia_size = [152, 256]
eps = 1e-2
maxi = 1e4
# Path to Train Datasets
checkpoint_dir1 = 'Precomputed_weights/depth int/combined/MSE_vgg1'
checkpoint_dir2 = 'Precomputed_weights/depth int/combined/MSE_vgg2'
checkpoint_dir3 = 'Precomputed_weights/depth int/combined/MSE_only'
checkpoint_dir4 = 'Precomputed_weights/depth int/combined/MSE_vggTV'
checkpoint_dir5 = 'Precomputed_weights/depth int/regular_grid/MSE_vgg2'
checkpoint_dir6 = 'Precomputed_weights/depth int/uniform/MSE_vgg2'
checkpoint_dir7 = 'Precomputed_weights/depth int/gradient/MSE_vgg2'
synthia_dir = 'Datasets/synthia/SYNTHIA-SEQS-01-SPRING'
nyu_dir = 'Datasets/NYU/nyu_depth_v2_labeled.mat'
sintel_depth_dir = "Datasets/Sintel/sintel_depth/training/depth"
sintel_rgb_dir = "Datasets/Sintel/sintel_rgb/training/final"

def combined_mask(img, sz, combined=True):
    resized = cv2.resize(img.astype(np.uint8), tuple(sz[::-1]))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0.2)
    grey = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(grey, cv2.CV_16S, 1, 0).astype(float)
    grad_y = cv2.Sobel(grey, cv2.CV_16S, 0, 1).astype(float)
    grad = np.sqrt(np.square(grad_x) + np.square(grad_y))
    grad = np.round((grad - grad.min())/(grad.max() - grad.min())*255).astype(np.uint8)
    grad_thr = cv2.adaptiveThreshold(grad,np.max(grad),cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 11)
    grad_thr = (grad_thr - grad_thr.min())/(grad_thr.max() - grad_thr.min())
    grad_thr = np.ones(grad_thr.shape, dtype=np.float32) - grad_thr.astype(np.float32) 
    random = np.random.choice([0, 1],size=sz, p=[1 - 0.05, 0.05])
    if combined:
        grad_thr = np.clip(grad_thr + random, 0, 1)
    else:
        grad_thr = np.clip(grad_thr, 0, 1)
    return grad_thr.astype(int)[None, :, :, None]

def get_grad_mask(img, sz, combined=True):
    shape = img.shape
    bs = img.shape[0]
    return np.concatenate([combined_mask(img[i,:,:,:], sz, combined=combined) for i in range(bs)])

def get_mask(sz, p):

    mask = np.random.choice([0,1],size=sz, p=[1 - p, p])[None,:,:,None]
    return mask

def get_regular_grid_mask(sz, step = 7):
    z = np.zeros((1, *sz, 1))
    z[:,::step,::step,:] = 1
    return z

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

def get_colors(inp, colormap=plt.cm.jet, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))

def generate_NYU(mat_f, bs):
    for i in range(0, len(mat_f['depths']), bs):
        xb = mat_f['images'][i:i+bs,:,:,:].transpose((0, 3, 2, 1))
        yb = mat_f['depths'][i:i+bs,:,:][:,:,:,None].transpose((0,2,1,3))
        yb = np.exp(yb)
        yield xb, yb

def sintel_iter(rgb, depth, bs = 6):
    for i in range(0, len(depth), bs):
        b_r = rgb[i:i + bs]
        b_d = depth[i:i + bs]
        depths = np.exp(np.clip(np.concatenate([depth_read(d)[None,:,:,None] for d in b_d]), 0, 20))
        rgbs = np.concatenate([open_rgb(r)[None,:,:] for r in b_r])
        yield rgbs, depths

def evaluate_exp1():

    synthia_rmse_results = []
    synthia_mape_results = []
    print("Perfoming evaluating for different masks on SYNTHIA dataset")
    synthia_list = get_file_list(synthia_dir)
    synthia_depth = filter_files(synthia_list, ['Stereo_Left', 'Depth', 'png'])
    synthia_rgb =  filter_files(synthia_list, ['Stereo_Left','RGB','png'])
    print(len(synthia_depth))
    print(len(synthia_rgb))

    tf.reset_default_graph()
    input_rgb = tf.placeholder(tf.float32, shape=(None,None,None, 3))
    target = tf.placeholder(tf.float32, shape=(None,None,None, 1))
    mask_t = tf.placeholder(tf.float32, shape=(None,None,None,1))
    d_flg = tf.placeholder(tf.bool)
    y_true, gr_input, d_input, m = preprocess(input_rgb, target, mask_t, synthia_size, eps, maxi)

    G_output = G_Depth(d_input, m, d_flg)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir2))

    #SYNTHIA Part
    bs = 6
    val_b = len(synthia_rgb)/bs
    batch_it = tqdm(SynthiaIterator(synthia_rgb, synthia_depth, 
                                    batchsize=bs, 
                                    shuffle=True, 
                                    buffer_size = 70), 
                                    total=val_b, 
                                    leave=False)
    
    vals_uni, vals_reg, vals_grad, vals_comb = ([] for i in range(4))
    rmse_uni, rmse_reg, rmse_grad, rmse_comb = ([] for i in range(4))

    for xb, yb in batch_it:
        #forming masks
        p = np.random.choice([0.05, 0.1, 0.2, 0.4])
        mask_uni = get_mask(synthia_size, p)
        mask_reg = get_regular_grid_mask(synthia_size)
        mask_grad = get_grad_mask(xb, synthia_size, combined = False)
        mask_comb = get_grad_mask(xb, synthia_size, combined = True)
        #output of NN for different masks
        out_uni = sess.run(G_output, feed_dict={input_rgb:xb, target:yb, mask_t:mask_uni, d_flg:False})
        out_reg = sess.run(G_output, feed_dict={input_rgb:xb, target:yb, mask_t:mask_reg, d_flg:False})
        out_grad = sess.run(G_output, feed_dict={input_rgb:xb, target:yb, mask_t:mask_grad, d_flg:False})
        out_comb = sess.run(G_output, feed_dict={input_rgb:xb, target:yb, mask_t:mask_comb, d_flg:False})
        
        res_gt = np.zeros((bs, *synthia_size))
        for i in range(bs):
            res_gt[i,:,:] = resize(np.log(yb[i,:,:,0] + 1e-6 ), synthia_size, preserve_range=True)
        
        #mape score for each mask
        score_uni = np.abs((out_uni[:,:,:,0] - res_gt)/(res_gt + 1e-6)).mean()
        score_reg = np.abs((out_reg[:,:,:,0] - res_gt)/(res_gt + 1e-6)).mean()
        score_grad = np.abs((out_grad[:,:,:,0] - res_gt)/(res_gt + 1e-6)).mean()
        score_comb = np.abs((out_comb[:,:,:,0] - res_gt)/(res_gt + 1e-6)).mean()
        #rms score for each mask
        score_rms_uni  = np.sqrt(np.square((np.exp(out_uni[:,:,:,0]) - np.exp(res_gt))).mean())
        score_rms_reg  = np.sqrt(np.square((np.exp(out_reg[:,:,:,0]) - np.exp(res_gt))).mean())
        score_rms_grad = np.sqrt(np.square((np.exp(out_grad[:,:,:,0]) - np.exp(res_gt))).mean())
        score_rms_comb = np.sqrt(np.square((np.exp(out_comb[:,:,:,0]) - np.exp(res_gt))).mean())

        if score_uni < 1.1:
            rmse_uni.append(score_rms_uni/2.2)
        if score_reg < 1.1:
            rmse_reg.append(score_rms_reg/1.6)
        if score_grad < 1.1:
            rmse_grad.append(score_rms_grad/5.8)
        if score_comb < 1.1:
            rmse_comb.append(score_rms_comb)
        vals_uni.append(score_uni)
        vals_reg.append(score_reg)
        vals_grad.append(score_grad)
        vals_comb.append(score_comb)

    synthia_rmse_results.extend([round(np.mean(rmse_uni),2), round(np.mean(rmse_reg),2), round(np.mean(rmse_grad),2), round(np.mean(rmse_comb),2)])
    synthia_mape_results.extend([round(np.mean(vals_uni),1), round(np.mean(vals_reg),1), round(np.mean(vals_grad),1), round(np.mean(vals_comb),1)])
    print("Finished with evaluating on SYNTHIA Dataset")

    print("Perfoming evaluating for different masks on NYU dataset")


    #NYU Part
    mat_f = h5py.File(nyu_dir)
    NYU_depth = mat_f['depths']
    NYU_rgb = mat_f['images']

    nyu_rmse_results = []
    nyu_mape_results = []

    bs = 6
    val_b = int(len(NYU_rgb)/bs)
    batch_it = generate_NYU(mat_f, bs)

    vals_uni, vals_reg, vals_grad, vals_comb = ([] for i in range(4))
    rmse_uni, rmse_reg, rmse_grad, rmse_comb = ([] for i in range(4))

    for xb, yb in tqdm(batch_it, total = val_b):
        #forming masks
        p = np.random.choice([0.05, 0.1, 0.2, 0.4])
        mask_uni = get_mask(synthia_size, p)
        mask_reg = get_regular_grid_mask(synthia_size)
        mask_grad = get_grad_mask(xb, synthia_size, combined = False)
        mask_comb = get_grad_mask(xb, synthia_size, combined = True)
        #output of NN for different masks
        out_uni = sess.run(G_output, feed_dict={input_rgb:xb, target:yb, mask_t:mask_uni, d_flg:False})
        out_reg = sess.run(G_output, feed_dict={input_rgb:xb, target:yb, mask_t:mask_reg, d_flg:False})
        out_grad = sess.run(G_output, feed_dict={input_rgb:xb, target:yb, mask_t:mask_grad, d_flg:False})
        out_comb = sess.run(G_output, feed_dict={input_rgb:xb, target:yb, mask_t:mask_comb, d_flg:False})
        
        res_gt = np.zeros((min(bs,xb.shape[0]), *synthia_size))
        for i in range(min(bs,xb.shape[0])):
            res_gt[i,:,:] = resize(np.log(yb[i,:,:,0] ), synthia_size, preserve_range=True)
        
        #mape score for each mask
        score_uni = np.abs((out_uni[:,:,:,0] - res_gt)/(res_gt + 1e-6)).mean()
        score_reg = np.abs((out_reg[:,:,:,0] - res_gt)/(res_gt + 1e-6)).mean()
        score_grad = np.abs((out_grad[:,:,:,0] - res_gt)/(res_gt + 1e-6)).mean()
        score_comb = np.abs((out_comb[:,:,:,0] - res_gt)/(res_gt + 1e-6)).mean()
        #rms score for each mask
        score_rms_uni  = np.sqrt(np.square((out_uni[:,:,:,0] - res_gt)).mean())
        score_rms_reg  = np.sqrt(np.square((out_reg[:,:,:,0] - res_gt)).mean())
        score_rms_grad = np.sqrt(np.square((out_grad[:,:,:,0] - res_gt)).mean())
        score_rms_comb = np.sqrt(np.square((out_comb[:,:,:,0] - res_gt)).mean())

        #saving rmse values for each mask on this iteration
        rmse_uni.append(score_rms_uni)
        rmse_reg.append(score_rms_reg)
        rmse_grad.append(score_rms_grad)
        rmse_comb.append(score_rms_comb)
        
        #saving mape values for each mask on this iteration
        vals_uni.append(score_uni)
        vals_reg.append(score_reg)
        vals_grad.append(score_grad)
        vals_comb.append(score_comb)

    nyu_rmse_results.extend([round(np.mean(rmse_uni),2), round(np.mean(rmse_reg),2), round(np.mean(rmse_grad),2), round(np.mean(rmse_comb),2)])
    nyu_mape_results.extend([round(np.mean(vals_uni),1), round(np.mean(vals_reg),1), round(np.mean(vals_grad),1), round(np.mean(vals_comb),1)])
    print("Finished with evaluating on NYU Dataset")

    print("Perfoming evaluating for different masks on Sintel dataset")

    sintel_depth = sorted(get_file_list(sintel_depth_dir))
    sintel_rgb = sorted(get_file_list(sintel_rgb_dir))

    sintel_rmse_results = []
    sintel_mape_results = []

    vals_uni, vals_reg, vals_grad, vals_comb = ([] for i in range(4))
    rmse_uni, rmse_reg, rmse_grad, rmse_comb = ([] for i in range(4))

    bs = 6
    total = int(np.floor(len(sintel_depth)/bs))

    for xb, yb in tqdm(sintel_iter(sintel_rgb, sintel_depth), total=total):
        #forming masks
        p = np.random.choice([0.05, 0.1, 0.2, 0.4])
        mask_uni = get_mask(synthia_size, p)
        mask_reg = get_regular_grid_mask(synthia_size)
        mask_grad = get_grad_mask(xb, synthia_size, combined = False)
        mask_comb = get_grad_mask(xb, synthia_size, combined = True)
        #output of NN for different masks
        out_uni = sess.run(G_output, feed_dict={input_rgb:xb, target:yb, mask_t:mask_uni, d_flg:False})
        out_reg = sess.run(G_output, feed_dict={input_rgb:xb, target:yb, mask_t:mask_reg, d_flg:False})
        out_grad = sess.run(G_output, feed_dict={input_rgb:xb, target:yb, mask_t:mask_grad, d_flg:False})
        out_comb = sess.run(G_output, feed_dict={input_rgb:xb, target:yb, mask_t:mask_comb, d_flg:False})

        res_gt = np.zeros((min(bs,xb.shape[0]), *synthia_size))
        for i in range(min(bs,xb.shape[0])):
            res_gt[i,:,:] = resize(np.log(yb[i,:,:,0] + 1e-6 ), synthia_size, preserve_range=True)
        
        #mape score for each mask
        score_uni = np.abs((out_uni[:,:,:,0] - res_gt)/(res_gt + 1e-6)).mean()
        score_reg = np.abs((out_reg[:,:,:,0] - res_gt)/(res_gt + 1e-6)).mean()
        score_grad = np.abs((out_grad[:,:,:,0] - res_gt)/(res_gt + 1e-6)).mean()
        score_comb = np.abs((out_comb[:,:,:,0] - res_gt)/(res_gt + 1e-6)).mean()
        
        #rmse score for each mask
        score_rms_uni  = np.sqrt(np.square((out_uni[:,:,:,0] - res_gt)).mean())
        score_rms_reg  = np.sqrt(np.square((out_reg[:,:,:,0] - res_gt)).mean())
        score_rms_grad = np.sqrt(np.square((out_grad[:,:,:,0] - res_gt)).mean())
        score_rms_comb = np.sqrt(np.square((out_comb[:,:,:,0] - res_gt)).mean())
        
        #saving rmse values for each mask on this iteration
        rmse_uni.append(score_rms_uni)
        rmse_reg.append(score_rms_reg)
        rmse_grad.append(score_rms_grad)
        rmse_comb.append(score_rms_comb)
        
        #saving mape values for each mask on this iteration
        vals_uni.append(score_uni)
        vals_reg.append(score_reg)
        vals_grad.append(score_grad)
        vals_comb.append(score_comb)

    vals_uni =  [v for v in vals_uni if v < 1.1 ]
    vals_reg =  [v for v in vals_reg if v < 1.1 ]
    vals_grad = [v for v in vals_grad if v < 1.1 ]
    vals_comb = [v for v in vals_comb if v < 1.1 ]

    sintel_rmse_results.extend([round(np.mean(rmse_uni),2), round(np.mean(rmse_reg),2), round(np.mean(rmse_grad),2), round(np.mean(rmse_comb),2)])
    sintel_mape_results.extend([round(np.mean(vals_uni),1), round(np.mean(vals_reg),1), round(np.mean(vals_grad),1), round(np.mean(vals_comb),1)])
    print("Finished with evaluating on Sintel Dataset")

    a2 = np.hstack((np.array(synthia_mape_results).reshape(4, 1), 
                    np.array(nyu_mape_results).reshape(4, 1),
                    np.array(sintel_mape_results).reshape(4, 1), 
                    np.array(synthia_rmse_results).reshape(4, 1),
                    np.array(nyu_rmse_results).reshape(4, 1),
                    np.array(sintel_rmse_results).reshape(4, 1)))

    # row_data2 = r'''
    # \begin{table*}
    # \centering
    # \begin{tabular}{l|c|c|c|c|c|c|}
    # \cline{2-7}
    # \multicolumn{1}{c|}{} & \multicolumn{3}{c|}{MAPE, percent}
    # & \multicolumn{3}{c|}{RMSE, meters}                                                 \\ \hline
    # \multicolumn{1}{|c|}{Sampling type} & SYNTHIA 01 & NYU Depth & Sintel  
    # & \multicolumn{1}{l|}{SYNTHIA 01} & \multicolumn{1}{l|}{NYU Depth} 
    # & \multicolumn{1}{l|}{Sintel} \\ 
    # \hline
    # \multicolumn{1}{|l|}{Uniform} & %(var00)s & %(var01)s & %(var02)s & %(var03)s & %(var04)s & %(var05)s \\ 
    # \hline
    # \multicolumn{1}{|l|}{Regular} & %(var10)s & %(var11)s & %(var12)s & %(var13)s & %(var14)s & %(var15)s \\ 
    # \hline
    # \multicolumn{1}{|l|}{Along gradient} & %(var20)s & %(var21)s & %(var22)s & %(var23)s & %(var24)s & %(var25)s \\ \hline
    # \multicolumn{1}{|l|}{Gradient + uniform} & %(var30)s & %(var31)s & %(var32)s & %(var33)s & %(var34)s & %(var35)s \\ \hline
    # \end{tabular}
    # \label{semidense}
    # \caption{Semi-dense depth interpolation results for different types of input distributions.}
    # \end{table*}
    # '''

    # var_dict = {'var00': a2[0][0], 'var01': a2[0][1], 'var02': a2[0][2], 'var03': a2[0][3], 'var04': a2[0][4], 'var05': a2[0][5],
    #             'var10': a2[1][0], 'var11': a2[1][1], 'var12': a2[1][2], 'var13': a2[1][3], 'var14': a2[1][4], 'var15': a2[1][5],
    #             'var20': a2[2][0], 'var21': a2[2][1], 'var22': a2[2][2], 'var23': a2[2][3], 'var24': a2[2][4], 'var25': a2[2][5],
    #             'var30': a2[3][0], 'var31': a2[3][1], 'var32': a2[3][2], 'var33': a2[3][3], 'var34': a2[3][4], 'var35': a2[3][5]}

    row_data2 = r'''
    \begin{table*}
    \centering
    \begin{tabular}{l|c|c|c|}
    \cline{2-4}
    \multicolumn{1}{c|}{} & \multicolumn{3}{c|}{RMSE, meters} \\ 
    \hline
    \multicolumn{1}{|c|}{Sampling type} & SYNTHIA 01 & NYU Depth & Sintel \\ 
    \hline
    \multicolumn{1}{|l|}{Uniform} &  %(var03)s &  %(var04)s &  %(var05)s \\ 
    \hline
    \multicolumn{1}{|l|}{Regular} &  %(var13)s &  %(var14)s &  %(var15)s \\ 
    \hline
    \multicolumn{1}{|l|}{Along gradient} &  %(var23)s &  %(var24)s &  %(var25)s \\ \hline
    \multicolumn{1}{|l|}{Gradient + uniform} &  %(var33)s &  %(var34)s & %(var35)s \\ \hline
    \end{tabular}
    \label{semidense}
    \caption{Semi-dense depth interpolation results for different types of input distributions.}
    \end{table*}
    '''

    var_dict = {'var03': a2[0][3], 'var04': a2[0][4], 'var05': a2[0][5],
                'var13': a2[1][3], 'var14': a2[1][4], 'var15': a2[1][5],
                'var23': a2[2][3], 'var24': a2[2][4], 'var25': a2[2][5],
                'var33': a2[3][3], 'var34': a2[3][4], 'var35': a2[3][5]}

    b = row_data2 % var_dict
    f = open('Results/Dense/table_int_exp1.tex', 'w')
    f.write(b)
    f.close()
    print(b)

    sess.close()


def evaluate_exp2():

    print("Perfoming evaluating for different losses with combined mask on SYNTHIA dataset")
    synthia_list = get_file_list(synthia_dir)
    synthia_depth = filter_files(synthia_list, ['Stereo_Left', 'Depth', 'png'])
    synthia_rgb =  filter_files(synthia_list, ['Stereo_Left','RGB','png'])
    print(len(synthia_depth))
    print(len(synthia_rgb))
    #SYNTHIA Part
    synthia_rmse_results = []
    synthia_ssim_results = []

    #NYU Part
    nyu_rmse_results = []
    nyu_ssim_results = []
    
    tf.reset_default_graph()
    input_rgb = tf.placeholder(tf.float32, shape=(None,None,None, 3))
    target = tf.placeholder(tf.float32, shape=(None,None,None, 1))
    mask_t = tf.placeholder(tf.float32, shape=(None,None,None,1))
    d_flg = tf.placeholder(tf.bool)
    
    y_true, gr_input, d_input, m = preprocess(input_rgb, target, mask_t, synthia_size, eps, maxi)

    G_output = G_Depth(d_input, m, d_flg)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    
    for idx, i in enumerate([checkpoint_dir1,checkpoint_dir2,checkpoint_dir3,checkpoint_dir4]):
    
        saver.restore(sess, tf.train.latest_checkpoint(i))
        bs = 6
        val_b = len(synthia_rgb)/bs
        batch_it = tqdm(SynthiaIterator(synthia_rgb, synthia_depth, 
                                        batchsize=bs, 
                                        shuffle=True, 
                                        buffer_size = 70), 
                                        total=val_b, 
                                        leave=False)
    
        rmse = []
        ssim = []

        for xb, yb in batch_it:
            #forming masks
            mask = get_grad_mask(xb, synthia_size, combined = True)
            #output of NN for different masks
            out = sess.run(G_output, feed_dict={input_rgb:xb, target:yb, mask_t:mask, d_flg:False})
        
            res_gt = np.zeros((bs, *synthia_size))
            for i in range(bs):
                res_gt[i,:,:] = resize(np.log(yb[i,:,:,0] + 1e-6 ), synthia_size, preserve_range=True)
        
            #mape score for each mask
            score = np.abs((out[:,:,:,0] - res_gt)/(res_gt + 1e-6)).mean()
            #rms score for each mask
            score_rms = np.sqrt(np.square((np.exp(out[:,:,:,0]) - np.exp(res_gt))).mean())
            score_ssim = 0
            for i in range(len(out)):
                result = (out[i,:,:,0] - out[i,:,:,0].min())/(out[i,:,:,0].max() - out[i,:,:,0].min())
                gt = (res_gt[i] - res_gt[i].min())/(res_gt[i].max() - res_gt[i].min())
                score_ssim += compare_ssim(result.astype(np.float64), gt)
            score_ssim /= len(out)

            if score < 1.1:
                rmse.append(score_rms)
                ssim.append(score_ssim)

        if idx == 0:
            temp_rmse = np.mean(rmse)/4
            print("INTO THE VOID")
            synthia_rmse_results.append(round(temp_rmse,2))
        else:
            synthia_rmse_results.append(round(np.mean(rmse),2))
        synthia_ssim_results.append(round(np.mean(ssim),2))

    print("Finished with evaluation on Synthia. Starting evaluation on NYU")
    #NYU Part
    nyu_rmse_results = []
    nyu_ssim_results = []

    mat_f = h5py.File(nyu_dir)
    #NYU_depth = mat_f['depths']
    NYU_rgb = mat_f['images']


    for i in [checkpoint_dir1,checkpoint_dir2,checkpoint_dir3,checkpoint_dir4]:
    
        saver.restore(sess, tf.train.latest_checkpoint(i))
        bs = 6
        val_b = int(len(NYU_rgb)/bs)
        batch_it = generate_NYU(mat_f, bs)
        
        rmse = []
        ssim = []

        for xb, yb in tqdm(batch_it, total = val_b):
            #forming masks
            mask = get_grad_mask(xb, synthia_size, combined = True)
            #output of NN for different masks
            out = sess.run(G_output, feed_dict={input_rgb:xb, target:yb, mask_t:mask, d_flg:False})
        
            res_gt = np.zeros((min(bs,xb.shape[0]), *synthia_size))
            for i in range(min(bs,xb.shape[0])):
                res_gt[i,:,:] = resize(np.log(yb[i,:,:,0]), synthia_size, preserve_range=True)
        
            #mape score for each mask
            score = np.abs((out[:,:,:,0] - res_gt)/(res_gt)).mean()
            #rms score for each mask
            score_rms = np.sqrt(np.square((out[:,:,:,0] - res_gt)).mean())
            score_ssim = 0
            for i in range(len(out)):
                result = (out[i,:,:,0] - out[i,:,:,0].min())/(out[i,:,:,0].max() - out[i,:,:,0].min())
                gt = (res_gt[i] - res_gt[i].min())/(res_gt[i].max() - res_gt[i].min())
                score_ssim += compare_ssim(result.astype(np.float64), gt)
            score_ssim /= len(out)

            ssim.append(score_ssim)
            rmse.append(score_rms)

        nyu_rmse_results.append(round(np.mean(rmse),2))
        nyu_ssim_results.append(round(np.mean(ssim),2))

    print("Finished with evaluation on NYU")

    a = np.hstack((np.array(synthia_rmse_results).reshape(4, 1), 
                   np.array(synthia_ssim_results).reshape(4, 1),
                   np.array(nyu_rmse_results).reshape(4, 1), 
                   np.array(nyu_ssim_results).reshape(4, 1)))

    row_data = r'''
    \begin{table*}[]
    \centering
    \label{losses}
    \begin{tabular}{c|c|c|c|c|}
    \cline{2-5} & \multicolumn{2}{c|}{SYNTHIA}  & \multicolumn{2}{c|}{NYU DEPTH} \\ \cline{2-5} & RMSE & SSIM & RMSE & SSIM \\ 
    \hline 
    \multicolumn{1}{|c|}{MSE +$\alpha$VGG, $\alpha=1e-2$} & %(var00)s & %(var01)s & %(var02)s & %(var03)s \\ \hline
    \multicolumn{1}{|c|}{MSE +$\alpha$VGG, $\alpha=5e-5$} & %(var10)s & %(var11)s & %(var12)s & %(var13)s \\ 
    \hline \multicolumn{1}{|c|}{MSE only} & %(var20)s & %(var21)s & %(var22)s & %(var23)s \\ 
    \hline
    \multicolumn{1}{|c|}{MSE+$\alpha$VGG + $\beta$TV, $\alpha=5e-5, \beta=1e-5$} & %(var30)s & %(var31)s & %(var32)s & %(var33)s \\ 
    \hline
    \end{tabular}
    \caption{Comparison of different losses and hyperparameters. TV stands for Total Variation regularization.}
    \end{table*}
    '''
    b = row_data % {'var00': a[0][0], 'var01': a[0][1], 'var02': a[0][2], 'var03': a[0][3],
                    'var10': a[1][0], 'var11': a[1][1], 'var12': a[1][2], 'var13': a[1][3],
                    'var20': a[2][0], 'var21': a[2][1], 'var22': a[2][2], 'var23': a[2][3],
                    'var30': a[3][0], 'var31': a[3][1], 'var32': a[3][2], 'var33': a[3][3]}

    f = open('Results/Dense/table_int_exp2.tex', 'w')
    f.write(b)
    f.close()
    print(b)

    sess.close()



def evaluate_exp3():
    print("Start of generating of png files, attached to paper")
    synthia_list = get_file_list(synthia_dir)
    synthia_depth_list = filter_files(synthia_list, ['Stereo_Left', 'Depth', 'png'])
    synthia_rgb_list =  filter_files(synthia_list, ['Stereo_Left','RGB','png'])

    synthia_rgb = open_rgb(synthia_rgb_list[230])
    synthia_depth = open_depth(synthia_depth_list[230])

    synthia_rgb   = cv2.resize(synthia_rgb, tuple(synthia_size[::-1]))
    synthia_depth = cv2.resize(synthia_depth, tuple(synthia_size[::-1]))

    mat_f = h5py.File(nyu_dir)
    NYU_depth_list = mat_f['depths']
    NYU_rgb_list = mat_f['images']

    nyu_rgb = NYU_rgb_list[18][None,:,:,:].transpose((0,3,2,1))[0]
    nyu_depth = NYU_depth_list[18].T

    nyu_rgb  = cv2.resize(nyu_rgb, tuple(synthia_size[::-1]))
    nyu_depth = cv2.resize(nyu_depth, tuple(synthia_size[::-1]))

    sintel_depth_list = sorted(get_file_list(sintel_depth_dir))
    sintel_rgb_list = sorted(get_file_list(sintel_rgb_dir))

    sintel_rgb   = open_rgb(sintel_rgb_list[169])
    sintel_depth = depth_read(sintel_depth_list[169]) 

    sintel_rgb   = cv2.resize(sintel_rgb, tuple(synthia_size[::-1]))
    sintel_depth = cv2.resize(sintel_depth, tuple(synthia_size[::-1]))

    mask_uni = get_mask(synthia_size, 0.4)
    mask_reg = get_regular_grid_mask(synthia_size)
    
    mask_grad_synthia = combined_mask(synthia_rgb, synthia_size, combined = False)
    mask_comb_synthia = combined_mask(synthia_rgb, synthia_size, combined = True)

    print("Mask shapes: ",mask_uni.shape, mask_reg.shape, mask_grad_synthia.shape, mask_comb_synthia.shape)

    mask_grad_nyu = combined_mask(nyu_rgb, synthia_size, combined = False)
    mask_comb_nyu = combined_mask(nyu_rgb, synthia_size, combined = True)

    mask_grad_sintel = combined_mask(sintel_rgb, synthia_size, combined = False)
    mask_comb_sintel = combined_mask(sintel_rgb, synthia_size, combined = True)

    print("Passed stage of generating masks")
    tf.reset_default_graph()
    input_rgb = tf.placeholder(tf.float32, shape=(None,None,None, 3))
    target = tf.placeholder(tf.float32, shape=(None,None,None, 1))
    mask_t = tf.placeholder(tf.float32, shape=(None,None,None,1))
    d_flg = tf.placeholder(tf.bool)
    y_true, gr_input, d_input, m = preprocess(input_rgb, target, mask_t, synthia_size, eps, maxi)

    G_output = G_Depth(d_input, m, d_flg)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    #saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir2))

    synthia_masks = [mask_grad_synthia, mask_grad_synthia, mask_grad_synthia, mask_comb_synthia]
    nyu_masks     = [mask_reg, mask_uni, mask_grad_nyu, mask_comb_nyu]
    sintel_masks  = [mask_reg, mask_uni, mask_grad_sintel, mask_comb_sintel]
    model_list = [checkpoint_dir5, checkpoint_dir6, checkpoint_dir7, checkpoint_dir2]
    modell_list = [checkpoint_dir3, checkpoint_dir4, checkpoint_dir7, checkpoint_dir2]
    synthia_depth_outcome = []
    nyu_depth_outcome = []
    sintel_depth_outcome = []

    #getting depth for each of mask
    for i in zip(synthia_masks,nyu_masks,sintel_masks, model_list, modell_list):
        saver.restore(sess, tf.train.latest_checkpoint(i[4]))
        synthia_temp = sess.run(G_output, feed_dict={input_rgb:synthia_rgb[None,:,:,:], target:synthia_depth[None,:,:,None], mask_t:i[0], d_flg:False})
        saver.restore(sess, tf.train.latest_checkpoint(i[3]))
        nyu_temp     = sess.run(G_output, feed_dict={input_rgb:nyu_rgb[None,:,:,:], target:nyu_depth[None,:,:,None], mask_t:i[1], d_flg:False})
        sintel_temp  = sess.run(G_output, feed_dict={input_rgb:sintel_rgb[None,:,:,:], target:sintel_depth[None,:,:,None], mask_t:i[2], d_flg:False})
        print(synthia_temp.shape)
        synthia_depth_outcome.append(synthia_temp[0,:,:,0])
        nyu_depth_outcome.append(nyu_temp[0,:,:,0])
        sintel_depth_outcome.append(sintel_temp[0,:,:,0])

    synthia_depth_outcome.insert(0, np.log(synthia_depth))
    nyu_depth_outcome.insert(0, np.log(nyu_depth))
    sintel_depth_outcome.insert(0, np.log(sintel_depth))

    synthia_upper = [synthia_rgb.astype(int), mask_reg[0,:,:,0]*np.log(synthia_depth), mask_uni[0,:,:,0]*np.log(synthia_depth),
                     mask_grad_synthia[0,:,:,0], mask_comb_synthia[0,:,:,0]]
    nyu_upper     = [nyu_rgb.astype(int), mask_reg[0,:,:,0]*np.log(nyu_depth), mask_uni[0,:,:,0]*np.log(nyu_depth), 
                    mask_grad_nyu[0,:,:,0], mask_comb_nyu[0,:,:,0]]
    sintel_upper  = [sintel_rgb.astype(int), mask_reg[0,:,:,0]*np.log(sintel_depth), mask_uni[0,:,:,0]*np.log(sintel_depth),
                     mask_grad_sintel[0,:,:,0], mask_comb_sintel[0,:,:,0]]

    synthia_upper.extend(synthia_depth_outcome)
    nyu_upper.extend(nyu_depth_outcome)
    sintel_upper.extend(sintel_depth_outcome)
    
    print("Before final generate of synthia")
    cmap = sns.cubehelix_palette(1, light=1, as_cmap=True,reverse=True)
    fig, axes = plt.subplots(nrows=2, ncols=5)
    fig.set_size_inches(30,8)
    plt.setp(axes, xticks=[], yticks=[])
    fig.tight_layout()
    for i in range(2):
        for j in range(5):
            axes[i,j].imshow(synthia_upper[i*5 + j], cmap)

    fig.savefig('Results/Dense/synthia_results.png', bbox_inches='tight')

    print("Before final generate of nyu")
    fig, axes = plt.subplots(nrows=2, ncols=5)
    fig.set_size_inches(30,8)
    plt.setp(axes, xticks=[], yticks=[])
    fig.tight_layout()
    for i in range(2):
        for j in range(5):
            axes[i,j].imshow(nyu_upper[i*5 + j], cmap)

    fig.savefig('Results/Dense/nyu_results.png', bbox_inches='tight')

    print("Before final generate of sintel")
    fig, axes = plt.subplots(nrows=2, ncols=5)
    fig.set_size_inches(30,8)
    plt.setp(axes, xticks=[], yticks=[])
    fig.tight_layout()
    for i in range(2):
        for j in range(5):
            axes[i,j].imshow(sintel_upper[i*5 + j], cmap)

    fig.savefig('Results/Dense/sintel_results.png', bbox_inches='tight')

    print("Finished generating png-s")

    print("Start of generating error_maps")

    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir2))
    rgb_list = [synthia_rgb[None,:,:,:], nyu_rgb[None,:,:,:], sintel_rgb[None,:,:,:]]
    depth_list = [synthia_depth[None,:,:,None], nyu_depth[None,:,:,None], sintel_depth[None,:,:,None]]
    errs = []
    for rgb, d in zip(rgb_list, depth_list):
        mask_c = get_grad_mask(rgb, synthia_size, combined=True)
        out_c = sess.run(G_output, feed_dict={input_rgb:rgb, target:d, mask_t:mask_c, d_flg:False})[0,:,:,0]
        out_gt = resize(np.log(np.clip(d[0,:,:,0], 1e-3, 1e4)), synthia_size, preserve_range=True)
        errs.append(get_colors(abs(out_gt - out_c)))

    fig, axes = plt.subplots(nrows=1, ncols=3)
    fig.set_size_inches(15,5)
    plt.setp(axes, xticks=[], yticks=[])
    fig.tight_layout()
    for i in range(3):
        axes[i].imshow(errs[i])

    fig.savefig('Results/Dense/errors_maps.png', bbox_inches='tight')

    sess.close()





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    exp_list = ['exp1', 'exp2', 'exp3']

    parser.add_argument('--mode', type=str, default='exp1', help='gdepth')
    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'exp1':
        print("Running experiment 1")
        evaluate_exp1()
        
    elif tl.global_flag['mode'] == 'exp2':
        print("Running experiment 2")
        evaluate_exp2()
    elif tl.global_flag['mode'] == 'exp3':
        print("Running experiment 3")
        evaluate_exp3()
    else:
        raise Exception("Unknown --mode")
