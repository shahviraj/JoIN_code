'''
Clean the Lumos face dataset by filtering out images that are highly tilted, or with excessive props.
For the valid files, apply alignment transformation on all image components.
'''
import copy

import PIL.Image
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import os
import cv2
from tqdm import tqdm
import dlib
from PIL import Image, ImageDraw
import numpy as np
import math
import torchvision
import scipy
import scipy.ndimage
import torchvision.transforms as transforms
import zipfile
import glob
import sys

cntst = int(sys.argv[1])

# Define the params
# main component would always be zeroth component
comps = [('shadow0','.exr'), ('specular0', '.jpg') , ('coat0','.jpg'), ('sheen0','.jpg'), ('sss0','.jpg')]
#discard = ['prop-4', 'prop-5', 'prop-6', 'prop-7', 'prop-8']
dir = '/home/virajs/work/datasets/lumos/'
outdir = '/sensei-fs/users/virajs/work/data/lumos2/'
outres = 256
predictor = dlib.shape_predictor("/sensei-fs/users/virajs/work/models/shape_predictor_68_face_landmarks.dat")
# threshold of eye to eye distnace. Only the images with dist greater than thres will be used.
thres = 90.0
cnt = int(1e5*(cntst))

for dirname in ['albedo', 'alsh', 'natural', 'specular', 'shading', 'shadow', 'alsh2']:
    if not os.path.exists(outdir+dirname):
        os.makedirs(outdir+dirname, exist_ok=True)


def transform_image(img, qsizef, quadf, transform_size, output_size, enable_padding=True):
    shrink = int(np.floor(qsizef / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad = quadf/ shrink
        qsize = qsizef/ shrink
    else:
        quad = quadf
        qsize = qsizef
    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    return img

def tonemap(img, mode='reinhard', scale=1.0):
    if mode == 'reinhard':
        return scale*img / (1.0 + scale*img)
    elif mode == 'srgb':
        return torch.where(img<0.0031308,img*12.92,(1.055*torch.pow(img,1/2.4))-0.055)
    else:
        raise NotImplementedError

def load_and_linearize_jpg(zipfname, imgname):
    with zipfile.ZipFile(zipfname).open(imgname, 'r') as f:
        im = cv2.cvtColor(np.array(Image.open(f)), cv2.COLOR_BGR2RGB)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    imt = transforms.ToTensor()(im)
    imt = torch.pow(imt, 2.4)
    #im = transforms.ToPILImage()(imt)
    return imt

def load_exr(zipfname, imgname):
    with zipfile.ZipFile(zipfname).open(imgname, 'r') as f:
        cv_img = cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8),
                              cv2.IMREAD_UNCHANGED)
    im = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return im

def run_alignment(img, zipf, cnt, predictor=predictor, outdir=outdir , comps = comps, output_size=outres, transform_size=4096, thres = thres):

    # first run some checks on the filename
    if int(img[-5]) > 3:
        #print('img discarded due to prop!')
        return None

    # open the albedo image
    with zipfile.ZipFile(zipf).open(img, 'r') as f:
        im_al = cv2.cvtColor(np.array(Image.open(f)), cv2.COLOR_BGR2RGB)
        im_al = cv2.cvtColor(im_al, cv2.COLOR_BGR2RGB)
    detector = dlib.get_frontal_face_detector()
    dets = detector(im_al, 1)
    if len(dets) < 1:
        #print("Face not detected, Image discarded!")
        return None

    for k, d in enumerate(dets):
        shape = predictor(im_al, d)

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)


    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise


    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    eyedist = np.linalg.norm(eye_to_eye)
    if  eyedist < thres:
        print("Eye to eye dist: ", np.linalg.norm(eye_to_eye), ' is below threshold, image discarded!')
        return None

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quadf = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsizef = np.hypot(*x) * 2


    # apply inverse tone-mapping to albedo images
    albedo_image = transforms.ToTensor()(Image.fromarray(im_al))
    albedo_image = torch.pow(albedo_image, 2.4)

    # Diffuse shading
    # load shadow0
    shadow_zipf = zipf.replace('albedo', 'shadow0').replace('jpg', 'exr')
    shadow_img = img.replace('albedo', 'shadow0').replace('jpg', 'exr')
    shadow_image = load_exr(shadow_zipf, shadow_img)

    shadow_image = transforms.ToTensor()(Image.fromarray((255.0*shadow_image).astype(np.uint8)))

    # load SSS
    sss_zipf = zipf.replace('albedo', 'sss0')
    sss_img = img.replace('albedo', 'sss0')
    sss_image = load_and_linearize_jpg(sss_zipf, sss_img)

    # We divide sss by albedo to get the corresponding "shading"
    sss_image = sss_image / torch.clamp(albedo_image, min= 1 / 255.0)

    # The diffuse shading image is the shadow plus the sss "shading"
    diffuse_shading_image = shadow_image + sss_image

    # SPECULAR
    sheen_zipf= zipf.replace('albedo', 'sheen0')
    sheen_img = img.replace('albedo', 'sheen0')
    sheen_image = load_and_linearize_jpg(sheen_zipf, sheen_img)


    specular_zipf = zipf.replace('albedo', 'specular0')
    specular_img = img.replace('albedo', 'specular0')
    specular_image = load_and_linearize_jpg(specular_zipf, specular_img)


    coat_zipf = zipf.replace('albedo', 'coat0')
    coat_img = img.replace('albedo', 'coat0')
    coat_image = load_and_linearize_jpg(coat_zipf, coat_img)

    specular_image = specular_image + sheen_image + coat_image


    final_image = albedo_image * diffuse_shading_image + specular_image

    final_image = tonemap(final_image, 'reinhard', 4.0)
    diffuse_shading_image = tonemap(diffuse_shading_image, 'reinhard', 2.0)
    specular_image = tonemap(specular_image, 'reinhard', 4.0)


    final_im = transforms.ToPILImage()(final_image)
    final_sh = transforms.ToPILImage()(diffuse_shading_image)

    final_spec = transforms.ToPILImage()(specular_image)
    #final_alsh = transforms.ToPILImage()(alsh_image)

    final_im = transform_image(final_im, copy.copy(qsizef), copy.copy(quadf), output_size=output_size,
                                      transform_size=transform_size, enable_padding=True)
    final_sh = transform_image(final_sh, copy.copy(qsizef), copy.copy(quadf), output_size=output_size,
                               transform_size=transform_size, enable_padding=True)
    final_spec = transform_image(final_spec, copy.copy(qsizef), copy.copy(quadf), output_size=output_size,
                               transform_size=transform_size, enable_padding=True)
    final_al = transform_image(Image.fromarray(im_al), copy.copy(qsizef), copy.copy(quadf), output_size=output_size,
                               transform_size=transform_size, enable_padding=True)

    final_im.save(f'{outdir}/natural/{cnt:08d}.jpg')
    final_sh.save(f'{outdir}/shading/{cnt:08d}.jpg')

    final_al.save(f'{outdir}/albedo/{cnt:08d}.jpg')
    final_spec.save(f'{outdir}/specular/{cnt:08d}.jpg')

    cnt = cnt + 1
    return cnt


allzipflist = sorted(glob.glob(dir + 'albedo/*.zip'))

st = int(cntst*8)
en = min(int((cntst+1)*8),len(allzipflist))

zipflist = allzipflist[st:en]

print("total zip files: ", len(zipflist))
for zipf in tqdm(zipflist):
    imglist = sorted([v for v in zipfile.ZipFile(zipf).namelist() if '.jpg' in v])
    print(f"total image files in {zipf}: ", len(imglist))
    for img in tqdm(imglist):
        ret = run_alignment(img, zipf, cnt, outdir=outdir , comps = comps, output_size=outres, thres = thres)
        if ret is not None:
            cnt = ret

        if cnt % 100 == 0:
            print("Images so far: ", cnt)