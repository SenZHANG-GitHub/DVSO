# Modified a little by Sen to generate the npy files for disp_left/right

# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import pdb
import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import cv2
import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument("--save_img_only", action="store_const", default=False, const=True)
parser.add_argument('--seq',                 type=str)
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--mode',                      type=str,   help='train or test', default='test')
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')

parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)

parser.add_argument('--input_height',              type=int,   help='input height', default=256)
parser.add_argument('--input_width',               type=int,   help='input width', default=512)


parser.add_argument('--model_name',                type=str,   help='model name', default='monodepth')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')


parser.add_argument('--batch_size',                type=int,   help='batch size', default=8)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
parser.add_argument('--do_stereo',                             help='if set, will train the stereo model', action='store_true')
parser.add_argument('--wrap_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
parser.add_argument('--use_deconv',                            help='if set, will use transposed convolutions', action='store_true')
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)

parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')

parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')

args = parser.parse_args()

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]             # disp[0]: disp of left img
    r_disp = np.fliplr(disp[1,:,:])  # disp[1]: disp of flipped left img
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def test(params):
    """Test function."""
    #################################
    
    # arg.mode must be "test"
    dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode, args.save_img_only)
    left  = dataloader.left_image_batch
    right = dataloader.right_image_batch # None
    
    model = MonodepthModel(params, args.mode, left, right)


    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    if args.checkpoint_path == "":
        raise ValueError("Error: args.checkpoint_path must be specified")
    restore_path = args.checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)
    
    num_test_samples = count_text_lines(args.filenames_file)
    
    print('now testing {} files'.format(num_test_samples))
    
    disparities_left = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_right = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_pp_left = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_pp_right = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    
    if args.output_directory == "":
        raise ValueError("Error: args.output_directory must be given")
    output_directory = args.output_directory
    

    for step in range(num_test_samples):
        disp = sess.run([model.disp_left_est[0], model.disp_right_est[0]])
        disp_left = disp[0]
        disp_right = disp[1]
        
        disparities_left[step] = disp_left[0].squeeze()
        disparities_right[step] = disp_right[0].squeeze()
        
        disparities_pp_left[step] = post_process_disparity(disp_left.squeeze())
        disparities_pp_right[step] = post_process_disparity(disp_right.squeeze())

    print('done.')
    
    np.save(output_directory + '/disparities_left.npy',    disparities_left)
    np.save(output_directory + '/disparities_right.npy',    disparities_right)
    np.save(output_directory + '/disparities_pp_left.npy', disparities_pp_left)
    np.save(output_directory + '/disparities_pp_right.npy', disparities_pp_right)
        
    print('done!')


def save_img(params):
    """Test function."""
    #################################
    
    # arg.mode must be "test"
    dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode, args.save_img_only)
    left  = dataloader.left_image_batch
    right = dataloader.right_image_batch # None
    
    model = GetImageModel(params, args.mode, left, right)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    
    num_test_samples = count_text_lines(args.filenames_file)
    
    print('now testing {} files'.format(num_test_samples))

    
    if args.output_directory == "":
        raise ValueError("Error: args.output_directory must be given")
    output_directory = "{}/image_2".format(args.output_directory)
    if os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
    os.mkdir(output_directory)
    
    
    print("=> Saving the downsampled images of seq {}".format(args.seq))
    for step in range(num_test_samples):
        img = sess.run([model.left])
        left_img = img[0][0].astype(np.uint8)
        path_img = "{}/{:06d}.jpg".format(output_directory, step)
        cv2.imwrite(path_img, left_img)
        
    print('done!')
    

def main(_):

    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        do_stereo=args.do_stereo,
        wrap_mode=args.wrap_mode,
        use_deconv=args.use_deconv,
        alpha_image_loss=args.alpha_image_loss,
        disp_gradient_loss_weight=args.disp_gradient_loss_weight,
        lr_loss_weight=args.lr_loss_weight,
        full_summary=args.full_summary)


    if args.mode == "test":
        if args.save_img_only:
            save_img(params)
        else:
            test(params)
    else:
        raise ValueError("Error: args.mode must be test!")

if __name__ == '__main__':
    tf.app.run()
