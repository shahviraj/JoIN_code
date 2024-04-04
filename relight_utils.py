# Author: Viraj Shah, virajs@adobe.com; vjshah3@illinois.edu
# Modified from
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from argparse import ArgumentParser
from time import perf_counter
import lpips
import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
#from focal_frequency_loss import FocalFrequencyLoss as FFL
import dnnlib
import legacy
from matplotlib import pyplot as plt
from elpips import elpips
import glob
from tqdm import tqdm
import cv2
import matplotlib as mpl
import faiss
mpl.rcParams['figure.dpi'] = 220
mpl.rcParams['font.size'] = 4
from mpl_toolkits.axes_grid1 import ImageGrid
from relight_utils import *
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('./torch_utils/')
sys.path.append('./torch_utils/ops/')

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def tonemap(img, mode, scale=1.0):
    if mode == 'reinhard':
        return (scale*img) / (1.0 + (scale*img))
    elif mode == 'srgb':
        return torch.where(img<0.0031308,img*12.92,(1.055*torch.pow(img,1/2.4))-0.055)
    else:
        raise NotImplementedError

def inv_tonemap(img, mode, scale=1.0):
    if mode == 'reinhard':
        img = torch.clip(img, 0.0, 0.999)
        return ((img) / (scale*(1.0 - img)))
    elif mode == 'srgb':
        return torch.where(img<0.04045,img/12.92,torch.pow((img+0.055)/1.055,2.4))
    else:
        raise NotImplementedError


def load_network(network_pkl):
    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)  # type: ignore
    return G

def load_real_image(opts,res):
    fname = opts.datadir + f'{opts.target_fnum:06d}.jpg'
    try:
        real_im = Image.open(fname).convert('RGB')
    except:
        real_im = Image.open(fname[:-4] + '.png').convert('RGB')
    real_im = transforms.Resize(res)(real_im)
    real_im = transforms.ToTensor()(real_im).unsqueeze(0)
    
    if opts.use_mask:
        mask_fname = opts.datadir + f'{opts.target_fnum:06d}_mask.jpg'
        try:
            mask_im = Image.open(mask_fname).convert('RGB')
        except:
            mask_im = Image.open(mask_fname[:-4] + '.png').convert('RGB')
        mask_im = transforms.Resize(res)(mask_im)
        opts.mask = transforms.ToTensor()(mask_im).unsqueeze(0).to(opts.device)
        opts.mask[opts.mask > 0.0] = 1.0
        
    return torch.clip(255.0 * real_im, 0, 255.0)

def load_lumos_img(datadir, target_fnum, imgtype):
    fname = datadir + f'/albedo/{target_fnum:08d}.jpg'

    if imgtype == 'shading':
        fname_diff = fname.replace('albedo', 'shading')
        diff_img = Image.open(fname_diff).convert('RGB')
        dtm = transforms.ToTensor()(diff_img).unsqueeze(0)
        raw_resized = inv_tonemap(dtm,'reinhard', 2.0)

    elif imgtype == 'albedo':
        al_img = Image.open(fname).convert('RGB')
        dtm = transforms.ToTensor()(al_img).unsqueeze(0)
        raw_resized = torch.pow(dtm, 2.4)

    elif imgtype == 'specular':
        fname_spec = fname.replace('albedo', 'specular')
        spec_img = Image.open(fname_spec).convert('RGB')
        dtm = transforms.ToTensor()(spec_img).unsqueeze(0)
        raw_resized = inv_tonemap(dtm, 'reinhard', 4.0)

    else:
        raise NotImplementedError

    im = torch.clip(255.0 * dtm, 0, 255).to(torch.uint8)
    return raw_resized, im

def load_materials_img(datadir, target_fnum, imgtype):
    if imgtype == 'shading':
        fname_diff_dir = datadir + f'/{target_fnum:06d}/diffuse_dir0001.exr'
        fname_diff_indir = datadir + f'/{target_fnum:06d}/diffuse_indir0001.exr'

        im_diff_dir = cv2.imread(fname_diff_dir, -1)
        im_diff_dir = cv2.cvtColor(im_diff_dir, cv2.COLOR_BGR2RGB)

        im_diff_indir = cv2.imread(fname_diff_indir, -1)
        im_diff_indir = cv2.cvtColor(im_diff_indir, cv2.COLOR_BGR2RGB)

        im_shading = im_diff_dir + im_diff_indir
        raw_resized = torch.tensor(im_shading, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) # 1xcxhxw tensor in float32 form

        dtm = tonemap(raw_resized, 'reinhard', 4.0)

    elif imgtype == 'albedo':
        fname_diff_color = datadir + f'/{target_fnum:06d}/diffuse_color0001.exr'
        im_albedo = cv2.imread(fname_diff_color, -1)
        im_albedo = cv2.cvtColor(im_albedo, cv2.COLOR_BGR2RGB)

        raw_resized = torch.tensor(im_albedo, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        dtm = tonemap(raw_resized, 'srgb')

    elif imgtype == 'specular':
        fname_dir = datadir + f'/{target_fnum:06d}/glossy_dir0001.exr'
        fname_indir = fname_dir.replace('glossy_dir', 'glossy_indir')

        im_gloss_dir = cv2.imread(fname_dir, -1)
        im_gloss_dir = cv2.cvtColor(im_gloss_dir, cv2.COLOR_BGR2RGB)

        im_gloss_indir = cv2.imread(fname_indir, -1)
        im_gloss_indir = cv2.cvtColor(im_gloss_indir, cv2.COLOR_BGR2RGB)

        im_gloss_shading = (im_gloss_dir + im_gloss_indir)

        fname_color = fname_dir.replace('glossy_dir', 'glossy_color')
        # print(fname_color)
        im_gloss_color = cv2.imread(fname_color, -1)
        im_gloss_color = cv2.cvtColor(im_gloss_color, cv2.COLOR_BGR2RGB)

        raw_resized = torch.tensor(im_gloss_color * im_gloss_shading, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        dtm = tonemap(raw_resized, 'reinhard', 4.0)

    else:
        raise NotImplementedError

    im = torch.clip(255.0 * dtm, 0, 255).to(torch.uint8)
    return raw_resized, im

def save_tensor_img(img, filename):
    '''
    image is expected to be [batch size x C x H x W]
    0 to 255 scaling is expected.
    '''
    pilim = PIL.Image.fromarray(img.to(torch.uint8).permute(0, 2, 3, 1).squeeze().numpy())
    pilim.save(filename)


def logprint(*args):
    print(*args)

def project(
        G_sh,
        G_al,
        G_sp,
        impath,
        opts,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
):
    if opts.knn_w != 0.0:
        #assert single_w == True
        assert opts.knn_path != ''
        assert opts.knn_path != None

    def compute_w_stats(G, w_avg_samples=10000):
        logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
        z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        w_samples  = []

        for i in range(w_avg_samples//10000):
            w_samples.append(G.mapping(torch.from_numpy(z_samples[i*10000: (i+1)*10000]).to(opts.device), None, truncation_psi=0.7))  # [N, L, C]
        w_samples = torch.cat(w_samples)
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]

        w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
        w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
        return w_avg, w_std, w_samples

    opt_vars = []

    if not opts.sh_fixed:
        assert target.shape[1:] == (G_sh.img_channels, G_sh.img_resolution, G_sh.img_resolution)
        G_sh = copy.deepcopy(G_sh).eval().requires_grad_(False).to(opts.device)  # type: ignore
        noise_bufs_sh = {name: buf for (name, buf) in G_sh.synthesis.named_buffers() if 'noise_const' in name}
        # build knn index
        if opts.knn_w != 0.0:
            if os.path.exists(opts.knn_path+f'_sh_trunc_{opts.w_avg_samples}_index.bin') and os.path.exists(opts.knn_path + f'_sh_trunc_{opts.w_avg_samples}.npy'):
                # load knn index
                index_sh = faiss.read_index(opts.knn_path + f'_sh_trunc_{opts.w_avg_samples}_index.bin')
                w_sh_samples = np.load(opts.knn_path + f'_sh_trunc_{opts.w_avg_samples}.npy')
                w_sh_avg = np.mean(w_sh_samples, axis=0, keepdims=True)  # [1, 1, C]
                w_sh_std = (np.sum((w_sh_samples - w_sh_avg) ** 2) / opts.w_avg_samples) ** 0.5
            else:
                # build knn index
                w_sh_avg, w_sh_std, w_sh_samples = compute_w_stats(G_sh, w_avg_samples=opts.w_avg_samples)
                res = faiss.StandardGpuResources()
                index_sh = faiss.IndexFlatL2(w_sh_samples.shape[-1])
                index_sh = faiss.index_cpu_to_gpu(res, 0, index_sh)
                #w_sh_samples = torch.tensor(w_sh_samples,device=device, dtype=torch.float32)
                index_sh.add(w_sh_samples.squeeze())
                #print(index_sh.total)
                save_index_sh = faiss.index_gpu_to_cpu(index_sh)
                faiss.write_index(save_index_sh, opts.knn_path+f'_sh_trunc_{opts.w_avg_samples}_index.bin')
                np.save( opts.knn_path + f'_sh_trunc_{opts.w_avg_samples}.npy',w_sh_samples)
        else:
            w_sh_avg, w_sh_std, w_sh_samples = compute_w_stats(G_sh, w_avg_samples=10000)

        if opts.init != 'None':
            try:
                init_file = torch.load(opts.init + f'/inference_w/{opts.imid:06d}.pt')
            except:
                init_file = torch.load(opts.init + f'/inference_w/{opts.imid:08d}.pt')
            if opts.single_w:
                w_sh_opt = init_file['latent'][0].unsqueeze(0).unsqueeze(0)  # dim: 1 x 1 x C
            else:
                w_sh_opt = init_file['latent'].unsqueeze(0) # dim: 1x latent_dim x 512
            w_sh_opt.requires_grad = True
            w_sh_opt.to(torch.float32).to(opts.device)
        else:
            if opts.single_w:
                w_sh_opt = torch.tensor(w_sh_avg, dtype=torch.float32, device=opts.device,
                                        requires_grad=True)  # pylint: disable=not-callable
            else:
                w_sh_opt = torch.tensor(w_sh_avg.repeat(G_sh.mapping.num_ws, axis=1), dtype=torch.float32,
                                        device=opts.device, requires_grad=True)  # pylint: disable=not-callable

        # print the init
        if opts.single_w:
            sh_init = G_sh.synthesis(w_sh_opt.repeat([1, G_sh.mapping.num_ws, 1]), noise_mode='const')
        else:
            sh_init = G_sh.synthesis(w_sh_opt, noise_mode='const')
        save_torch_image((sh_init + 1.0) / 2.0, impath + '_shading_init.jpg')

        w_sh_out = torch.zeros([opts.num_steps + 1] + list(w_sh_opt.shape[1:]), dtype=torch.float32, device=opts.device)
        w_sh_out[0] = w_sh_opt.detach()[0]
        opt_vars = opt_vars + [w_sh_opt] + list(noise_bufs_sh.values())

        for buf in list(noise_bufs_sh.values()):
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

    if not opts.al_fixed:
        assert target.shape[1:] == (G_al.img_channels, G_al.img_resolution, G_al.img_resolution)
        G_al = copy.deepcopy(G_al).eval().requires_grad_(False).to(opts.device)  # type: ignore
        noise_bufs_al = {name: buf for (name, buf) in G_al.synthesis.named_buffers() if 'noise_const' in name}

        if opts.knn_w != 0.0:
            if os.path.exists(opts.knn_path+f'_al_trunc_{opts.w_avg_samples}_index.bin') and os.path.exists(opts.knn_path + f'_al_trunc_{opts.w_avg_samples}.npy'):
                # load knn index
                index_al = faiss.read_index(opts.knn_path + f'_al_trunc_{opts.w_avg_samples}_index.bin')
                w_al_samples = np.load(opts.knn_path + f'_al_trunc_{opts.w_avg_samples}.npy')
                w_al_avg = np.mean(w_al_samples, axis=0, keepdims=True)  # [1, 1, C]
                w_al_std = (np.sum((w_al_samples - w_al_avg) ** 2) / opts.w_avg_samples) ** 0.5
            else:
                # build knn index
                w_al_avg, w_al_std, w_al_samples = compute_w_stats(G_al, w_avg_samples=opts.w_avg_samples)
                res = faiss.StandardGpuResources()
                index_al = faiss.IndexFlatL2(w_al_samples.shape[-1])
                index_al = faiss.index_cpu_to_gpu(res, 0, index_al)
                index_al.add(w_al_samples.squeeze())
                save_index_al = faiss.index_gpu_to_cpu(index_al)
                faiss.write_index(save_index_al, opts.knn_path+f'_al_trunc_{opts.w_avg_samples}_index.bin')
                np.save( opts.knn_path +f'_al_trunc_{opts.w_avg_samples}.npy', w_al_samples)
        else:
            w_al_avg, w_al_std, w_al_samples = compute_w_stats(G_al, w_avg_samples=10000)
        if opts.init != 'None':
            # if opts.dataset == 'lumos':
                # init_file = torch.load(opts.init.replace('sh', 'al') + f'/inference_w/{opts.imid:08d}.pt')
            try:
                init_file = torch.load(opts.init.replace('-sh-', '-al-') + f'/inference_w/{opts.imid:06d}.pt')
            except:
                init_file = torch.load(opts.init.replace('-sh-', '-al-') + f'/inference_w/{opts.imid:08d}.pt')
            if opts.single_w:
                w_al_opt = init_file['latent'][0].unsqueeze(0).unsqueeze(0)  # dim: 1 x 1 x C
            else:
                w_al_opt = init_file['latent'].unsqueeze(0)  # dim: 1x latent_dim x 512
            w_al_opt.requires_grad = True
            w_al_opt.to(torch.float32).to(opts.device)
        else:
            if opts.single_w:
                w_al_opt = torch.tensor(w_al_avg, dtype=torch.float32, device=opts.device,
                                        requires_grad=True)  # pylint: disable=not-callable
            else:
                w_al_opt = torch.tensor(w_al_avg.repeat(G_al.mapping.num_ws, axis=1), dtype=torch.float32,
                                        device=opts.device, requires_grad=True)  # pylint: disable=not-callable
        # print the init
        if opts.single_w:
            al_init = G_al.synthesis(w_al_opt.repeat([1, G_al.mapping.num_ws, 1]), noise_mode='const')
        else:
            al_init = G_al.synthesis(w_al_opt, noise_mode='const')
        save_torch_image((al_init + 1.0) / 2.0, impath + '_albedo_init.jpg')

        w_al_out = torch.zeros([opts.num_steps + 1] + list(w_al_opt.shape[1:]), dtype=torch.float32, device=opts.device)
        w_al_out[0] = w_al_opt.detach()[0]
        opt_vars = opt_vars + [w_al_opt] + list(noise_bufs_al.values())

        for buf in list(noise_bufs_al.values()):
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

    if opts.use_specular:
        assert target.shape[1:] == (G_sp.img_channels, G_sp.img_resolution, G_sp.img_resolution)
        G_sp = copy.deepcopy(G_sp).eval().requires_grad_(False).to(opts.device)  # type: ignore
        noise_bufs_sp = {name: buf for (name, buf) in G_sp.synthesis.named_buffers() if 'noise_const' in name}
        # build knn index
        if opts.knn_w != 0.0:
            if os.path.exists(opts.knn_path+f'_sp_trunc_{opts.w_avg_samples}_index.bin') and os.path.exists(opts.knn_path + f'_sp_trunc_{opts.w_avg_samples}.npy'):
                # load knn index
                index_sp = faiss.read_index(opts.knn_path + f'_sp_trunc_{opts.w_avg_samples}_index.bin')
                w_sp_samples = np.load(opts.knn_path + f'_sp_trunc_{opts.w_avg_samples}.npy')
                w_sp_avg = np.mean(w_sp_samples, axis=0, keepdims=True)  # [1, 1, C]
                w_sp_std = (np.sum((w_sp_samples - w_sp_avg) ** 2) / opts.w_avg_samples) ** 0.5
            else:
                # build knn index
                w_sp_avg, w_sp_std, w_sp_samples = compute_w_stats(G_sp, w_avg_samples=opts.w_avg_samples)
                res = faiss.StandardGpuResources()
                index_sp = faiss.IndexFlatL2(w_sp_samples.shape[-1])
                index_sp = faiss.index_cpu_to_gpu(res, 0, index_sp)
                #w_sp_samples = torch.tensor(w_sp_samples,device=device, dtype=torch.float32)
                index_sp.add(w_sp_samples.squeeze())
                #print(index_sp.total)
                save_index_sp = faiss.index_gpu_to_cpu(index_sp)
                faiss.write_index(save_index_sp, opts.knn_path+f'_sp_trunc_{opts.w_avg_samples}_index.bin')
                np.save( opts.knn_path + f'_sp_trunc_{opts.w_avg_samples}.npy',w_sp_samples)
        else:
            w_sp_avg, w_sp_std, w_sp_samples = compute_w_stats(G_sp, w_avg_samples=10000)

        if opts.init != 'None':
            try:
                init_file = torch.load(opts.init.replace('-sh-', '-sp-') + f'/inference_w/{opts.imid:06d}.pt')
            except:
                init_file = torch.load(opts.init.replace('-sh-', '-sp-') + f'/inference_w/{opts.imid:08d}.pt')
            if opts.single_w:
                w_sp_opt = init_file['latent'][0].unsqueeze(0).unsqueeze(0)  # dim: 1 x 1 x C
            else:
                w_sp_opt = init_file['latent'].unsqueeze(0)  # dim: 1x latent_dim x 512
            w_sp_opt.requires_grad = True
            w_sp_opt.to(torch.float32).to(opts.device)
        else:
            if opts.single_w:
                w_sp_opt = torch.tensor(w_sp_avg, dtype=torch.float32, device=opts.device,
                                        requires_grad=True)  # pylint: disable=not-callable
            else:
                w_sp_opt = torch.tensor(w_sp_avg.repeat(G_sp.mapping.num_ws, axis=1), dtype=torch.float32,
                                        device=opts.device, requires_grad=True)  # pylint: disable=not-callable

        # print the init
        if opts.single_w:
            sh_init = G_sp.synthesis(w_sp_opt.repeat([1, G_sp.mapping.num_ws, 1]), noise_mode='const')
        else:
            sh_init = G_sp.synthesis(w_sp_opt, noise_mode='const')
        save_torch_image((sh_init + 1.0) / 2.0, impath + '_specular_init.jpg')

        w_sp_out = torch.zeros([opts.num_steps + 1] + list(w_sp_opt.shape[1:]), dtype=torch.float32, device=opts.device)
        w_sp_out[0] = w_sp_opt.detach()[0]
        opt_vars = opt_vars + [w_sp_opt] + list(noise_bufs_sp.values())

        for buf in list(noise_bufs_sp.values()):
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(opts.device)

    # Features for target image.
    target_images = target
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    if opts.loss_fn == 'vgg' or opts.loss_fn == 'both':
        target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    optimizer = torch.optim.Adam(opt_vars, betas=(0.9, 0.999), lr=opts.initial_learning_rate)
    # Init noise.

    if opts.loss_fn == 'lpips':
        lpips_loss = lpips.LPIPS(net='vgg').to(opts.device)
    elif opts.loss_fn == 'elpips':
        elpips_loss = elpips.ELPIPS().to(opts.device)

    all_loss = [0.0]
    all_dist = [0.0]
    all_lr = [0.0]
    for step in tqdm(range(opts.num_steps)):
        # Learning rate schedule.
        t = step / opts.num_steps
        if not opts.sh_fixed:
            w_noise_scale_sh = w_sh_std * opts.initial_noise_factor * max(0.0, 1.0 - t / opts.noise_ramp_length) ** 2
        if not opts.al_fixed:
            w_noise_scale_al = w_al_std * opts.initial_noise_factor * max(0.0, 1.0 - t / opts.noise_ramp_length) ** 2
        if opts.use_specular:
            w_noise_scale_sp = w_sp_std * opts.initial_noise_factor * max(0.0, 1.0 - t / opts.noise_ramp_length) ** 2

        lr_ramp = min(1.0, (1.0 - t) / opts.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / opts.lr_rampup_length)
        lr = opts.initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        all_lr.append(lr)

        # Synth images from opt_w
        if not opts.sh_fixed:
            w_sh_noise = torch.randn_like(w_sh_opt) * w_noise_scale_sh
            ws_sh = (w_sh_opt) + (w_sh_noise)  # .repeat([1, G.mapping.num_ws, 1])

            if opts.single_w:
                synth_sh_images = G_sh.synthesis(ws_sh.repeat([1, G_sh.mapping.num_ws, 1]), noise_mode='const')
            else:
                synth_sh_images = G_sh.synthesis(ws_sh, noise_mode='const')
            synth_sh_images = gan_to_raw(synth_sh_images, 'shading', opts)
        else:
            # add pre/post processing steps
            synth_sh_images = G_sh

        # Synth images from opt_w
        if not opts.al_fixed:
            w_al_noise = torch.randn_like(w_al_opt) * w_noise_scale_al
            ws_al = (w_al_opt) + (w_al_noise)  # .repeat([1, G.mapping.num_ws, 1])

            if opts.single_w:
                synth_al_images = G_al.synthesis(ws_al.repeat([1, G_al.mapping.num_ws, 1]), noise_mode='const')
            else:
                synth_al_images = G_al.synthesis(ws_al, noise_mode='const')
            synth_al_images = gan_to_raw(synth_al_images, 'albedo', opts)
        else:
            # add pre/post processing steps
            synth_al_images = G_al

            # Synth images from opt_w
        if opts.use_specular:
            w_sp_noise = torch.randn_like(w_sp_opt) * w_noise_scale_sp
            ws_sp = (w_sp_opt) + (w_sp_noise)  # .repeat([1, G.mapping.num_ws, 1])

            if opts.single_w:
                synth_sp_images = G_sp.synthesis(ws_sp.repeat([1, G_sp.mapping.num_ws, 1]), noise_mode='const')
            else:
                synth_sp_images = G_sp.synthesis(ws_sp, noise_mode='const')
            synth_sp_images = gan_to_raw(synth_sp_images, 'specular', opts)
        else:
            # add pre/post processing steps
            synth_sp_images = G_sp
        _, synth_im_images = run_forward_model(synth_al_images, synth_sh_images, synth_sp_images, opts)

        if synth_im_images.shape[2] > 256:
            synth_im_images = F.interpolate(synth_im_images, size=(256, 256), mode='area')

        if opts.loss_fn == 'vgg':
            synth_features = vgg16(synth_im_images, resize_images=False, return_lpips=True)
            dist = (target_features - synth_features).square().sum()
        elif opts.loss_fn == 'l2':
            dist = (target_images - synth_im_images).square().mean()
        elif opts.loss_fn == 'lpips':
            rescaled_target_images = (target_images / 127.5) - 1.0
            rescaled_synth_images = (synth_im_images / 127.5) - 1.0
            dist = lpips_loss(rescaled_target_images, rescaled_synth_images)
        elif opts.loss_fn == 'elpips':
            rescaled_target_images = (target_images / 127.5) - 1.0
            rescaled_synth_images = (synth_im_images / 127.5) - 1.0
            dist = elpips_loss(rescaled_target_images, rescaled_synth_images)
        else:
            raise NotImplementedError

        # Noise regularization.
        reg_loss = 0.0
        in_domain_loss = 0.0
        knn_loss = 0.0
        if not opts.sh_fixed:
            for v in list(noise_bufs_sh.values()):
                noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2) * 2.0
            in_domain_loss += (ws_sh - torch.tensor(w_sh_avg, device=opts.device)).square().mean()
            if opts.knn_w != 0.0:
                with torch.no_grad():
                    distnn, sh_indices = index_sh.search(ws_sh.detach().cpu().numpy()[0,...], 50)
                    w_sh_neigh = np.mean(w_sh_samples[sh_indices, 0, :], 1, keepdims=False)[np.newaxis, ...]
                knn_loss = knn_loss + (((ws_sh - torch.tensor(w_sh_neigh, device=opts.device, requires_grad=False)) / (opts.sh_rad* 0.5 * ws_sh.shape[1])).square().sum() - 1.0).clip(min=0.0)
                #print("sh_knn_loss: ", (((ws_sh - torch.tensor(w_sh_neigh, device=opts.device, requires_grad=False)) / (opts.sh_rad * 0.5 * ws_sh.shape[1])).square().sum() - 1.0).item())
                #print("org_sh_dist: ", distnn)
        if not opts.al_fixed:
            for v in list(noise_bufs_al.values()):
                noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2) * 2.0
            in_domain_loss += (ws_al - torch.tensor(w_al_avg, device=opts.device)).square().mean()
            if opts.knn_w != 0.0:
                with torch.no_grad():
                    distnn, al_indices = index_al.search(ws_al.detach().cpu().numpy()[0,...], 50)
                    w_al_neigh = np.mean(w_al_samples[al_indices, 0, :], 1, keepdims=False)[np.newaxis, ...]
                knn_loss = knn_loss + (((ws_al - torch.tensor(w_al_neigh, device=opts.device, requires_grad=False)) / (opts.al_rad * 0.5* ws_al.shape[1])).square().sum() - 1.0).clip(min=0.0)
                #print("al_knn_loss: ", (((ws_al - torch.tensor(w_al_neigh, device=opts.device, requires_grad=False)) / (opts.al_rad* 0.5* ws_al.shape[1])).square().sum() - 1.0).item())

        if opts.use_specular:
            for v in list(noise_bufs_sp.values()):
                noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2) * 2.0
            in_domain_loss += (ws_sp - torch.tensor(w_sp_avg, device=opts.device)).square().mean()
            if opts.knn_w != 0.0:
                with torch.no_grad():
                    distnn, sh_indices = index_sp.search(ws_sp.detach().cpu().numpy()[0,...], 50)
                    w_sp_neigh = np.mean(w_sp_samples[sh_indices, 0, :], 1, keepdims=False)[np.newaxis, ...]
                knn_loss = knn_loss + (((ws_sp - torch.tensor(w_sp_neigh, device=opts.device, requires_grad=False)) / (opts.sp_rad * 0.5 * ws_sp.shape[1])).square().sum() - 1.0).clip(min=0.0)
                #print("sp_knn_loss: ", (((ws_sp - torch.tensor(w_sp_neigh, device=opts.device, requires_grad=False)) / (opts.sp_rad * 0.5 * ws_sp.shape[1])).square().sum() - 1.0).item())
                #print("org_sh_dist: ", distnn)

        loss = dist + reg_loss * opts.regularize_noise_weight + in_domain_loss * opts.in_domain_w + knn_loss * opts.knn_w

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        all_dist.append(dist.item())
        all_loss.append(loss.item())
        if step%100 == 0:
            if opts.knn_w != 0:
                logprint(
                f'step {step + 1:>4d}/{opts.num_steps}: dist {dist.item():<4.4f} total loss {float(loss.item()):<5.4f} in domain loss {float(opts.in_domain_w * in_domain_loss.item()):<5.6f} knn loss {float(opts.knn_w * knn_loss.item()):<5.6f}')
            else:
                logprint(
                    f'step {step + 1:>4d}/{opts.num_steps}: dist {dist.item():<4.4f} total loss {float(loss.item()):<5.4f} in domain loss {float(opts.in_domain_w * in_domain_loss.item()):<5.6f} ')

        # Save projected W for each optimization step.
        if not opts.sh_fixed:
            w_sh_out[step + 1] = ws_sh.detach()[0]
        if not opts.al_fixed:
            w_al_out[step + 1] = ws_al.detach()[0]
        if opts.use_specular:
            w_sp_out[step + 1] = ws_sp.detach()[0]

    if not opts.sh_fixed:
        projected_w_sh = w_sh_out[-1]
        if opts.single_w:
            synth_sh_image = G_sh.synthesis(projected_w_sh.unsqueeze(0).repeat([1, G_sh.mapping.num_ws, 1]),
                                            noise_mode='const')
        else:
            synth_sh_image = G_sh.synthesis(projected_w_sh.unsqueeze(0), noise_mode='const')
        synth_sh_raw_image = gan_to_raw(synth_sh_image, 'shading', opts)
        final_sh_image = gan_to_final(synth_sh_image, opts)
        final_sh_image.save(f'{impath}_final_shading.jpg')
        np.savez(f'{impath}_shading.npz', w=projected_w_sh.unsqueeze(0).cpu().numpy())
        # plot_next_img(final_sh_pilimage, "Recovered Shading")
    else:
        synth_sh_raw_image = G_sh
        final_sh_image = None

    if not opts.al_fixed:
        projected_w_al = w_al_out[-1]
        if opts.single_w:
            synth_al_image = G_al.synthesis(projected_w_al.unsqueeze(0).repeat([1, G_al.mapping.num_ws, 1]),
                                            noise_mode='const')
        else:
            synth_al_image = G_al.synthesis(projected_w_al.unsqueeze(0), noise_mode='const')
        synth_al_raw_image = gan_to_raw(synth_al_image, 'albedo', opts)
        final_al_image = gan_to_final(synth_al_image, opts)
        np.savez(f'{impath}_albedo.npz', w=projected_w_al.unsqueeze(0).cpu().numpy())

        # plot_next_img(final_al_pilimage, "Recovered Albedo")
        final_al_image.save(f'{impath}_final_albedo.jpg')
    else:
        synth_al_raw_image = G_al
        final_al_image = None

    if opts.use_specular:
        projected_w_sp = w_sp_out[-1]
        if opts.single_w:
            synth_sp_image = G_sp.synthesis(projected_w_sp.unsqueeze(0).repeat([1, G_sp.mapping.num_ws, 1]),
                                            noise_mode='const')
        else:
            synth_sp_image = G_sp.synthesis(projected_w_sp.unsqueeze(0), noise_mode='const')
        synth_sp_raw_image = gan_to_raw(synth_sp_image, 'specular', opts)
        final_sp_image = gan_to_final(synth_sp_image, opts)
        final_sp_image.save(f'{impath}_final_specular.jpg')
        np.savez(f'{impath}_specular.npz', w=projected_w_sp.unsqueeze(0).cpu().numpy())
        # plot_next_img(final_sp_pilimage, "Recovered Shading")
    else:
        synth_sp_raw_image = G_sp
        final_sp_image = None

    _, final_target_image = run_forward_model(synth_al_raw_image, synth_sh_raw_image, synth_sp_raw_image, opts)
    final_target_image = final_target_image.permute(0, 2, 3, 1).clamp(0, 255.0).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(final_target_image, 'RGB').save(f'{impath}_projection.jpg')

    return PIL.Image.fromarray(final_target_image, 'RGB'), final_sh_image, final_al_image, final_sp_image 

# ----------------------------------------------------------------------------
def gan_to_final(img,opts):
    img = (img + 1.0) * (1.0 / 2)
    if opts.use_mask:
        img = opts.mask*img
    final_img = (255.0 * img).permute(0, 2, 3, 1).clamp(0, 255.0).to(torch.uint8)[0].cpu().numpy()
    final_img = PIL.Image.fromarray(final_img, 'RGB')
    return final_img

def gan_to_raw(img, comptype, opts):
    img = (img + 1.0) * (1.0 / 2)
    img = torch.clip(img, 0.01, 0.99)
    if opts.dataset == 'materials':
        if comptype == 'albedo':
            img = inv_tonemap(img, 'srgb', 1.0)
        if comptype == 'shading':
            img = inv_tonemap(img, 'reinhard', 4.0)
        if comptype == 'specular':
            img = inv_tonemap(img, 'reinhard', 4.0)
    if opts.dataset == 'lumos':
        if comptype == 'albedo':
            img = inv_tonemap(img, 'srgb', 1.0)
        if comptype == 'shading':
            img = inv_tonemap(img, 'reinhard', 2.0)
        if comptype == 'specular':
            img = inv_tonemap(img, 'reinhard', 4.0)
    if opts.use_mask:
        img = opts.mask*img
    
    return img

def run_forward_model(al, sh, sp, opts):
    if opts.dataset == 'materials':
        targetraw = (sh * al) + sp
        targetim = tonemap(targetraw, 'reinhard', 4.0)
        targetim = torch.clip(255.0 * targetim, 0, 255.0)
    elif opts.dataset == 'lumos':
        targetraw = (sh * al) + sp
        targetim = tonemap(targetraw, 'reinhard', 4.0)
        targetim = torch.clip(255.0 * targetim, 0, 255.0)
    else:
        raise NotImplementedError

    return targetraw, targetim  # return in float32 form.

def prepare_images(opts):
    if opts.dataset == 'materials':
        shadingraw, shadingim = load_materials_img(opts.datadir, opts.target_fnum, 'shading')
        albedoraw, albedoim = load_materials_img(opts.datadir, opts.target_fnum, 'albedo')
        specularraw, specularim = load_materials_img(opts.datadir, opts.target_fnum, 'specular')

        targetraw, targetim = run_forward_model(albedoraw, shadingraw, specularraw, opts)

        # return format: torch tensors of 1xcxhxw, clipped between 0 to 255. comps are in uint8 and target is in float32
        return albedoraw, albedoim, shadingraw, shadingim,specularraw, specularim, targetraw, targetim
    elif opts.dataset == 'lumos':
        shadingraw, shadingim = load_lumos_img(opts.datadir, opts.target_fnum, 'shading')
        albedoraw, albedoim = load_lumos_img(opts.datadir, opts.target_fnum, 'albedo')
        specularraw, specularim = load_lumos_img(opts.datadir, opts.target_fnum, 'specular')

        targetraw, targetim = run_forward_model(albedoraw, shadingraw, specularraw, opts)
        
        # return format: torch tensors of 1xcxhxw, clipped between 0 to 255. comps are in uint8 and target is in float32
        return albedoraw, albedoim, shadingraw, shadingim, specularraw, specularim, targetraw, targetim
        #return load_lumos_images(opts)
    else:
        raise NotImplementedError

def run_metric(fim, fsh, fal, fsp, tarim, tarsh, taral, tarsp, opts, pti=False):

    def convert_img(img, device=opts.device):
        return (torch.from_numpy(np.array(img)) / 127.5 - 1.0).to(device).permute(2, 0, 1).unsqueeze(0)

    def convert_tensor(img, device=opts.device):
        return (img.to(torch.float32).to(device))/127.5 - 1.0

    import lpips
    loss_fn_alex = lpips.LPIPS(net='alex').to(opts.device)

    def lpipsi(img1, img2):
        return loss_fn_alex(img1, img2).item()

    from torchmetrics import MeanSquaredError
    mse = MeanSquaredError().to(opts.device)

    def l2(img1, img2):
        return mse(img1, img2).item()

    from torchmetrics import StructuralSimilarityIndexMeasure
    ms_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(opts.device)

    def ssim(img1, img2):
        return ms_ssim(img1, img2).item()

    from torchmetrics import PeakSignalNoiseRatio
    mpsnr = PeakSignalNoiseRatio().to(opts.device)

    def psnr(img1, img2):
        return mpsnr(img1, img2).item()

    metriclist = ['l2', 'psnr', 'ssim', 'lpips']
    met_fns = {
        'l2': l2,
        'psnr': psnr,
        'ssim': ssim,
        'lpips': lpipsi
    }

    if not opts.real_image:
        tarsh = convert_tensor(tarsh)
        taral = convert_tensor(taral)
        tarsp = convert_tensor(tarsp)
    tarim = convert_tensor(tarim)

    fim = convert_img(fim)
    if not opts.real_image:
        fsh = convert_img(fsh)
        fal = convert_img(fal)
        fsp = convert_img(fsp)

    for j, met in enumerate(metriclist):
        if os.path.exists(opts.outdir + f'/{met}_metric.pt'):
            metric_dict = torch.load(opts.outdir + f'_pti/{met}_metric.pt')
        else:
            metric_dict = {}


        met_fn = met_fns[met]
        if not opts.real_image:
            metric_dict[opts.imid] = {'albedo': None, 'shading': None, 'specular': None, 'image': None}
            if not opts.sh_fixed:
                metric_dict[opts.imid]['shading'] = met_fn(tarsh, fsh)
            if not opts.al_fixed:
                metric_dict[opts.imid]['albedo'] = met_fn(taral, fal)
            if opts.use_specular:
                metric_dict[opts.imid]['specular'] = met_fn(tarsp, fsp)
            metric_dict[opts.imid]['image'] = met_fn(tarim, fim)
        else:
            metric_dict[opts.imid] = {'image': None}
            metric_dict[opts.imid]['image'] = met_fn(tarim, fim)

        torch.save(metric_dict, opts.outdir + f'_pti/{met}_metric.pt')


# python relight-projector.py --datadir=/home/virajs/work/data/hypersim/ --outdir=/home/virajs/work/outputs/projections/both_albedo_shading/ --target=69123 --network1=/home/virajs/work/outputs/training_runs/00000-illum_train_256_63k-paper256-batch64/network-snapshot-020966.pkl --network2=/home/virajs/work/models/reflect-network-snapshot-016128.pkl --alpha=10000 --albedo-fixed=False --shading-fixed=False
def update_opts(opts):
    opts.device = torch.device('cuda')
    opts.target_fnum = int(opts.target_fnum)
    opts.imid = opts.target_fnum
    opts.initial_learning_rate = opts.lr
    opts.initial_noise_factor = 0.0
    opts.lr_rampdown_length = 0.25
    opts.lr_rampup_length = 0.05
    opts.noise_ramp_length = 0.75
    opts.regularize_noise_weight = opts.alpha
    opts.w_avg_samples = 100000
    if opts.dataset == 'materials':
        opts.sh_rad = 14.57
        opts.al_rad = 15.30
        opts.sp_rad = 14.60
    elif opts.dataset == 'lumos':
        opts.sh_rad = 7.23
        opts.al_rad = 5.33
        opts.sp_rad = 6.09
    else:
        raise NotImplementedError
    return opts

def run_projection(opts):
    """Project given image to the latent space of pretrained network pickle.
    """
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{opts.gpu_id}"
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)

    opts = update_opts(opts)

    os.makedirs(opts.outdir, exist_ok=True)
    impath = f'{opts.outdir}/{opts.imid}_knn_w_{opts.knn_w}_id_w_{opts.in_domain_w}_{opts.alpha}_lr_{opts.lr}_loss_{opts.loss_fn}_sh_{opts.sh_fixed}_al_{opts.al_fixed}_sp_{opts.use_specular}_data_{opts.dataset}'

    if not opts.real_image:
        alr,alim, shr,shim, spr,spim, tarr, tarim = prepare_images(opts) # return form is torch tensors of 1xcxhxw, clipped between 0 to 255. comps are in uint8 and target is in float32

        # Load networks.
        if opts.sh_fixed:
            G_sh = shr.to(opts.device)
        else:
            G_sh = load_network(opts.network_pkl_shading)

        if opts.al_fixed:
            G_al = alr.to(opts.device)
        else:
            G_al = load_network(opts.network_pkl_albedo)

        if not opts.use_specular:
            G_sp = torch.zeros_like(spr).to(opts.device) # if specular is not used, then set it to zero (blank image)
        else:
            G_sp = load_network(opts.network_pkl_specular)


        target_pil = PIL.Image.fromarray(tarim.to(torch.uint8).permute(0, 2, 3, 1).squeeze().numpy())

        target_pil.save(f'{opts.outdir}/target_{opts.imid}.jpg')
        save_tensor_img(shim, f'{opts.outdir}/target_shading_{opts.imid}.jpg')
        save_tensor_img(alim, f'{opts.outdir}/target_albedo_{opts.imid}.jpg')
        save_tensor_img(spim, f'{opts.outdir}/target_specular_{opts.imid}.jpg')

    else:
        assert not opts.al_fixed
        assert not opts.sh_fixed
        G_sh = load_network(opts.network_pkl_shading)
        G_al = load_network(opts.network_pkl_albedo)
        tarim = load_real_image(opts, res=G_sh.img_resolution)
        if not opts.use_specular:
            G_sp = torch.zeros_like(tarim).to(opts.device) # if specular is not used, then set it to zero (blank image)
        else:
            G_sp = load_network(opts.network_pkl_specular)
        target_pil = PIL.Image.fromarray(tarim.to(torch.uint8).permute(0, 2, 3, 1).squeeze().numpy())

        target_pil.save(f'{opts.outdir}/target_{opts.imid}.jpg')
    # Optimize projection.
    start_time = perf_counter()

    final_target_image, final_sh_image, final_al_image, final_sp_image = project(
        G_sh,
        G_al,
        G_sp,
        impath,
        opts,
        target=tarim.to(opts.device)
    )

    if not opts.real_image:
        run_metric(final_target_image, final_sh_image, final_al_image, final_sp_image, tarim, shim, alim, spim, opts)
    else:
        run_metric(final_target_image, final_sh_image, final_al_image, final_sp_image, tarim, tarsh=None, taral=None, tarsp=None, opts=opts)
    print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

    
    
    
    

def plot_results(img_id, dirname, save=False, spec=True, real=False, pti=False):
    
    def get_f(pat):
        if 'target' in pat:
            print(sorted(glob.glob(f"{dirname}/{pat}_{img_id}*"))[0])
            return sorted(glob.glob(f"{dirname}/{pat}_{img_id}*"))[0]
        else:
            print(sorted(glob.glob(f"{dirname}/{img_id}_*{pat}*"))[0])
            return sorted(glob.glob(f"{dirname}/{img_id}_*{pat}*"))[0]
    def get_pti_f(pat):
        if 'target' in pat:
            print(sorted(glob.glob(f"{dirname_pti}/{pat}_{img_id}*"))[0])
            return sorted(glob.glob(f"{dirname_pti}/{pat}_{img_id}*"))[0]
        else:
            print(sorted(glob.glob(f"{dirname_pti}/{img_id}_*{pat}*"))[0])
            return sorted(glob.glob(f"{dirname_pti}/{img_id}_*{pat}*"))[0]

    plt.figsize=(6,4)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=1.1,
                    wspace=0.4,
                    hspace=0.4)
    
    if pti:
        dirname_pti = dirname + '_pti'
        nc = 4
        nr = 4
    else:
        nc = 3
        nr = 4
    
    
    plt.subplot(nr,nc,1)
    plt.imshow(Image.open(get_f('target')))
    plt.axis('off')
    plt.title("Target")
    
    if not real:
        plt.subplot(nr,nc,1+nc)
        plt.imshow(Image.open(get_f('target_shading')))
        plt.axis('off')
        plt.title("shading")

        plt.subplot(nr,nc,1+2*nc)
        plt.imshow(Image.open(get_f('target_albedo')))
        plt.axis('off')
        plt.title("albedo")
        
        if spec:
            plt.subplot(nr,nc,1+3*nc)
            plt.imshow(Image.open(get_f('target_specular')))
            plt.axis('off')
            plt.title("specular")

    plt.subplot(nr,nc,2)
    plt.imshow(Image.open(get_f('projection')))
    plt.axis('off')
    plt.title("Opt. Inversion")

    plt.subplot(nr,nc,2+nc)
    plt.imshow(Image.open(get_f('final_shading')))
    plt.axis('off')
    plt.title("shading")

    plt.subplot(nr,nc,2+2*nc)
    plt.imshow(Image.open(get_f('final_albedo')))
    plt.axis('off')
    plt.title("albedo")

    if spec:
        plt.subplot(nr,nc,2+3*nc)
        plt.imshow(Image.open(get_f('final_specular')))
        plt.axis('off')
        plt.title("specular")

    # plt.subplot(4,3,2)
    # plt.imshow(Image.open(f[5]))
    # plt.axis('off')
    # plt.title("Init")

    plt.subplot(nr,nc,3+nc)
    plt.imshow(Image.open(get_f('shading_init')))
    plt.axis('off')
    plt.title("shading init")

    plt.subplot(nr,nc,3+2*nc)
    plt.imshow(Image.open(get_f('albedo_init')))
    plt.axis('off')
    plt.title("albedo init")

    if spec:
        plt.subplot(nr,nc,3+3*nc)
        plt.imshow(Image.open(get_f('specular_init')))
        plt.axis('off')
        plt.title("specular init")
    
    if pti:
        plt.subplot(nr,nc,4)
        plt.imshow(Image.open(get_pti_f('projection_pti')))
        plt.axis('off')
        plt.title("PTI Inversion")
    
        plt.subplot(nr,nc,4+nc)
        plt.imshow(Image.open(get_pti_f('shading_pti')))
        plt.axis('off')
        plt.title("shading PTI")

        plt.subplot(nr,nc,4+2*nc)
        plt.imshow(Image.open(get_pti_f('albedo_pti')))
        plt.axis('off')
        plt.title("albedo PTI")

        if spec:
            plt.subplot(nr,nc,4+3*nc)
            plt.imshow(Image.open(get_pti_f('specular_pti')))
            plt.axis('off')
            plt.title("specular PTI")
    
    #plt.show()
    
    if save:
        os.makedirs(dirname+'_grid' , exist_ok=True)
        plt.savefig(dirname+f'_grid/{img_id}_all.jpg', bbox_inches='tight')

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--network1', dest='network_pkl_shading', help='Network pickle filename for shading generator',required=True)
    parser.add_argument('--network2', dest='network_pkl_albedo', help='Network pickle filename for albedo generator',required=True)
    parser.add_argument('--network3', dest='network_pkl_specular', help='Network pickle filename for specular generator',required=True)
    parser.add_argument('--init', dest='init', help='location of init ws code file (.pt file)', default='None')
    parser.add_argument('--target', dest='target_fnum', help='Target image file number to project to', required=True)
    parser.add_argument('--datadir', help='location of the data directory', required=True)
    parser.add_argument('--num-steps', help='Number of optimization steps', type=int, default=1000, )
    parser.add_argument('--alpha', help='Regularize noise weight', type=float, default=1e5)
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-1)
    parser.add_argument('--in-domain-w', help='regularizing wt for in domain loss', type=float, default=0.0)
    parser.add_argument('--knn-w', help='regularizing wt for in domain loss by using knn method', type=float, default=0.0)
    parser.add_argument('--knn-path', help='path to the knn index for knn-based in domain loss', required=False)
    parser.add_argument('--seed', help='Random seed', type=int, default=303)
    parser.add_argument('--gpu-id', help='gpu id', type=int, default=0)
    parser.add_argument('--single-w', help='use single w for optimization',action='store_true')
    parser.add_argument('--real-image', help='inversion on real image -- no groundtruth is available', action='store_true')
    parser.add_argument('--sh-fixed', help='Keep shading fixed and change only albedo', action='store_true')
    parser.add_argument('--al-fixed', help='Keep albedo fixed and change only shading', action='store_true')
    parser.add_argument('--use-specular', help='Do not decompose into specular. Just decompose into albedo and shading', action='store_true')
    parser.add_argument('--outdir', help='Where to save the output images', required=True)
    parser.add_argument('--dataset', help='name of the dataset', required=True)
    parser.add_argument('--loss-fn', help='loss function to use; l2 or vgg', required=False, default='elpips')

    opts = parser.parse_args()
    run_projection(opts)  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
