#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import shutil
import torch

import numpy as np
import read_write_binary as im

import subprocess
# cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
#
# os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    if not os.path.exists(gts_path):
        os.makedirs(gts_path)

    # debug = 0
    t_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize(); t0 = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize(); t1 = time.time()

        t_list.append(t1-t0)

        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{}'.format(view.image_name) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{}'.format(view.image_name) + ".png"))

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # ------------------------------------------------------
    if os.path.exists(args.source_path):
        shutil.rmtree(args.source_path)
        os.mkdir(args.source_path)
    else:
        os.mkdir(args.source_path)

    shutil.copytree(os.path.join(args.source_path, "../sparse"), os.path.join(args.source_path, "gt_sparse"))
    shutil.copytree(os.path.join(args.source_path, "../images"), os.path.join(args.source_path, "images"))

    shutil.copy(os.path.join(args.source_path, "../test_aligned_pose.txt"),
                os.path.join(args.source_path, "test_aligned_pose.txt"))
    data = im.read_images_binary(os.path.join(args.source_path, "gt_sparse", "images.bin"))
    #  若读取的是原训练集的sfm位姿文件，则image 恒定为 data[1]；若上面的sparse文件是已存在的，则需要根据不同仿真场景变化不同的起始idx
    image = data[1]
    new_data = {}
    with open(os.path.join(args.source_path, "test_aligned_pose.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            n, tx, ty, tz, qx, qy, qz, qw = line.split(" ")
            name = "{}.png".format(n)
            if not os.path.exists(os.path.join(args.source_path, "images", name)):
                images = [i for i in os.listdir(os.path.join(args.source_path, "images")) if ".png" in i]
                shutil.copy(os.path.join(os.path.join(args.source_path, "images", images[0])),
                            os.path.join(os.path.join(args.source_path, "images", name)))
            i = int(n)
            qvec = [float(i) for i in [qw, qx, qy, qz]]
            tvec = [float(i) for i in [tx, ty, tz]]
            #image = data[1]
            image = image._replace(id=i, qvec=np.array(qvec), tvec=np.array(tvec), name=name)
            #data[1 + i] = image
            new_data[i] = image
    print(len(new_data))
    if not os.path.exists(os.path.join(args.source_path, "gt_sparse/0")):
        os.mkdir(os.path.join(args.source_path, "gt_sparse/0"))
    im.write_images_binary(new_data, os.path.join(args.source_path, "gt_sparse/0", "images.bin"))
    # -------------------------------------------------------------------------------------------------------

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
