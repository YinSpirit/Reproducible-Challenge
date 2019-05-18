#! /usr/bin/python
# -*- coding: utf8 -*-
import time
import tensorlayer as tl
import progressbar
import zipfile
import os

import urllib.request

pbar = None

def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

def download_synthia(seq):
    
    url_softrain = "http://synthia-dataset.net/download/757/"
    url_summer = "http://synthia-dataset.net/download/797/"
    url_spring = "http://synthia-dataset.net/download/733/"
    url_spring1 = "http://synthia-dataset.net/download/743/"

    if seq == '02':
        url = url_softrain
    elif seq == '05':
        url = url_spring
    elif seq == '01':
        url = url_spring1
    else:
        url = url_summer

    if seq == '02' or seq == '06':
        url_to_place = "Datasets/synthia/train"
    else:
        url_to_place = "Datasets/synthia"

    print(f"Downloading {seq}-sequence of synthia dataset:")

    urllib.request.urlretrieve(url, 'Datasets/synthia/seq.zip', show_progress)
    print("Finished with downloading. Starting Unzipping:")
    try:
        with zipfile.ZipFile(f"Datasets/synthia/seq.zip", allowZip64=True) as zip_ref:
            zip_ref.extractall(url_to_place)

        print("Finished unzipping. Starting the deleting of zip file")
        os.remove(f"{url_to_place}/seq.zip")
        print("Finished deleting")
    except:
        print("There is problem with unziping. Unzipping should be made manually")

def download_sintel():
    url_depth = "http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip"
    url_png = "http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_images.zip"
    print("Downloading sintel depth files:")
    urllib.request.urlretrieve(url_depth, 'Datasets/depth_sintel.zip', show_progress)
    print("Finished with depth files. Starting downloading of png files:")
    urllib.request.urlretrieve(url_png, 'Datasets/rgb_sintel.zip', show_progress)
    print("Finished Download. Starting unzipping of the depth_sintel.zip:")
    try:
        with zipfile.ZipFile("Datasets/depth_sintel.zip","r") as zip_ref:
            zip_ref.extractall("Datasets/sintel/sintel_depth")
        print("Finished unzipping of depth_sintel.zip. Starting unzipping of the depth_rgb.zip")
        with zipfile.ZipFile("Datasets/rgb_sintel.zip","r") as zip_ref:
            zip_ref.extractall("Datasets/sintel/sintel_rgb")
        print("Finished unzipping of rgb_sintel.zip. Deleting zip files")
        os.remove("Datasets/rgb_sintel.zip")
        os.remove("Datasets/depth_sintel.zip")
        print("Finished deleting. Sintel dataset is located in Datasets/sintel folder")
    except: 
        print("There is problem with unziping. Unzipping should be made manually")

def download_nyu():
    url = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    urllib.request.urlretrieve(url, 'Datasets/nyu/nyu_depth_v2_labeled.mat', show_progress)
    print("Finished Download. NYU dataset is located in Datasets/nyu folder")

def download_middleburry():
    url = "http://vision.middlebury.edu/stereo/data/scenes2005/ThirdSize/zip-2views/ALL-2views.zip"
    urllib.request.urlretrieve(url, 'Datasets/middle.zip', show_progress)
    print("Finished Download. Starting unzipping of the file:")
    with zipfile.ZipFile("Datasets/middle.zip","r") as zip_ref:
        zip_ref.extractall("Datasets/middlebury")
    os.remove("Datasets/middle.zip")
    print("Finished Unzipping. Middlebury dataset is located in Datasets/Middlebury folder")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='nyu', help='gdepth')
    parser.add_argument('--seq', type=str, default='02', help='seq')
    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    tl.global_flag['seq'] = args.seq

    if tl.global_flag['mode'] == 'synthia':
        print(f"Downloading Synthia dataset - {tl.global_flag['seq']} sequence")
        download_synthia(tl.global_flag['seq'])
    elif tl.global_flag['mode'] == 'nyu':
        print("Downloading Nyu dataset")
        download_nyu()
    elif tl.global_flag['mode'] == 'test':
        print("testing of zip")
        test()
    elif tl.global_flag['mode'] == 'middleburry':
        print("Downloading Middleburry dataset")
        download_middleburry()
    elif tl.global_flag['mode'] == 'sintel':
        print("Downloading Sintel dataset")
        download_sintel()
    else:
        raise Exception("Unknow --mode")
