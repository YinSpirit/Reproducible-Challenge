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

def download_vgg():
    url = "https://www.dropbox.com/s/7mmianmwcj2qyl5/vgg16.npy?dl=1"
    urllib.request.urlretrieve(url, 'vgg/vgg16.npy', show_progress)

def download_models():
    url = "https://www.dropbox.com/s/f51f795qq7of9rt/Reproducible%20challenge.zip?dl=1"
    urllib.request.urlretrieve(url, 'Precomputed_weights.zip', show_progress)
    print("Finished Download. Starting unzipping of the file:")
    with zipfile.ZipFile("Precomputed_weights.zip","r") as zip_ref:
        zip_ref.extractall("Precomputed_weights")
    os.remove("Precomputed_weights.zip")
    print("Finished Unzipping. Weights of the model located in Precomputed_weights folder")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='vgg', help='gdepth')
    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'vgg':
        print("Downloading vgg-16 weights")
        download_vgg()
    elif tl.global_flag['mode'] == 'weights':
        print("Downloading precomputed weights for each model")
        download_models()
        #pass
    else:
        raise Exception("Unknow --mode")
