"""
This is Sen's naive implementation of forward warping!
-> The float disparities are simply floored to int!
"""


import numpy as np
import pdb
import os
import argparse
import sys
from tqdm import tqdm

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=str, required=True)
    parser.add_argument("--disp_path", type=str, 
                        default="data/kitti/odometry/monodepth2_disps")
    return parser.parse_args()

def readCalib(seq, disp_path):
    lines = []
    with open("{}/{}/camera_small.txt".format(disp_path, seq), mode="r") as f:
        lines = f.readlines()
    fx = float(lines[0].strip().split()[0])
    w = int(lines[3].strip().split()[0])
    h = int(lines[3].strip().split()[1])
    bl = float(lines[4].strip())
    return fx, w, h, bl

def check(opt):
    """
    Check the consistency of the warped right disparity and the predicted right disparity using monodepth 
    -> opt.disp_path should be data/kitti/odometry/monodepth_disps
    """
    fx, w, h, bl = readCalib(opt.seq, opt.disp_path)
    disp_left = np.load("{}/{}/disparities_pp_left.npy".format(opt.disp_path, opt.seq))
    disp_right = np.load("{}/{}/disparities_pp_right.npy".format(opt.disp_path, opt.seq))


    diff_means = [] # the mean value of diff maps
    diff_medians = [] # the median value of diff maps 
    diff_1_percents = [] # percentage of pixels with disp diff <= 1 
    diff_2_percents = [] # percentage of pixels with disp diff <= 2
    missing_percents = [] # percentange of missing pixels in the warped disp

    for idx in tqdm(range(disp_left.shape[0])):
        tmp_left = disp_left[idx] * w
        tmp_right = disp_right[idx] * w

        virtual_right = np.zeros((h, w))
        # virtual_right_rev = np.zeros((h, w))

        for iw in range(w):
            for ih in range(h):
                new_w = iw + int(tmp_left[ih, iw])
                if 0 <= new_w < w:
                    if virtual_right[ih, new_w] > 0:
                        # If multiple-pixel collisions happen, we use the closer one to relieve occlusion
                        if abs(tmp_left[ih, iw]) > abs(virtual_right[ih, new_w]):
                            virtual_right[ih, new_w] = tmp_left[ih, iw] 
                    else:
                        virtual_right[ih, new_w] = tmp_left[ih, iw] 
                    # virtual_right_rev[ih, new_w] = tmp_left[ih, iw] * -1

        # By comparing with tmp_right, virtual_right is correct without times -1
        mask = virtual_right > 0
        diff = abs(mask * (virtual_right - tmp_right))
        missing_percent = 1 - mask.flatten().sum() / (w * h)
        diff_1_percent = 1 - (diff > 1).flatten().sum() / (w * h)
        diff_2_percent = 1 - (diff > 2).flatten().sum() / (w * h)
        
        diff_means.append(diff.mean())
        diff_medians.append(np.median(diff))
        diff_1_percents.append(diff_1_percent)
        diff_2_percents.append(diff_2_percent)
        missing_percents.append(missing_percent)

    pdb.set_trace()


def warp_right_disp(opt):
    """Warp the left disparity to a warped right disparity using monodepth2 left_disp
    -> opt.disp_path should be data/kitti/odometry/monodepth2_disps
    -> usage: python warp_right_disp.py --seq 09 --disp_path data/kitti/odometry/monodepth2_disps
    """
    if opt.disp_path != "data/kitti/odometry/monodepth2_disps":
        raise ValueError("Error: --disp_path must be data/kitti/odometry/monodepth2_disps")
    
    fx, w, h, bl = readCalib(opt.seq, opt.disp_path)
    disp_left = np.load("{}/{}/disparities_pp_left.npy".format(opt.disp_path, opt.seq))
    disp_right = np.zeros(disp_left.shape, dtype=disp_left.dtype)

    for idx in tqdm(range(disp_left.shape[0])):
        tmp_left = disp_left[idx] * w

        virtual_right = np.zeros((h, w))

        for iw in range(w):
            for ih in range(h):
                new_w = iw + int(tmp_left[ih, iw])
                if 0 <= new_w < w:
                    if virtual_right[ih, new_w] > 0:
                        # If multiple-pixel collisions happen, we use the closer one to relieve occlusion
                        if abs(tmp_left[ih, iw]) > abs(virtual_right[ih, new_w]):
                            virtual_right[ih, new_w] = tmp_left[ih, iw] 
                    else:
                        virtual_right[ih, new_w] = tmp_left[ih, iw] 
                    # virtual_right_rev[ih, new_w] = tmp_left[ih, iw] * -1

        # By comparing with tmp_right, virtual_right is correct without times -1
        mask = virtual_right > 0
        disp_right[idx] = virtual_right / w
    
    out_path = "{}/{}/disparities_pp_right.npy".format(opt.disp_path, opt.seq)
    print("=> Saving warped results to {}".format(out_path))
    np.save(out_path, disp_right)


if __name__ == "__main__":
    opt = parseArgs()
    warp_right_disp(opt)




