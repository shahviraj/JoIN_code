# Author: Viraj Shah, virajs@adobe.com; vjshah3@illinois.edu
'''
This code is used to prepare the results for the paper and the presentation, by uploading them in the correct format, and
generating the latex strings for the figures.
'''

import os.path
import numpy as np
from PIL import Image
import glob
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 220
mpl.rcParams['font.size'] = 4
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm


def plot_results(img_id, dirname):
    def get_f(pat):
        if 'target' in pat:
            print(sorted(glob.glob(f"{dirname}/{pat}_{img_id}*"))[0])
            return sorted(glob.glob(f"{dirname}/{pat}_{img_id}*"))[0]
        else:
            print(sorted(glob.glob(f"{dirname}/{img_id}_*{pat}*"))[0])
            return sorted(glob.glob(f"{dirname}/{img_id}_*{pat}*"))[0]

    plt.figsize = (6, 4)
    plt.subplot(4, 3, 1)
    plt.imshow(Image.open(get_f('target')))
    plt.axis('off')
    plt.title("Target")

    plt.subplot(4, 3, 4)
    plt.imshow(Image.open(get_f('target_shading')))
    plt.axis('off')
    plt.title("shading")

    plt.subplot(4, 3, 7)
    plt.imshow(Image.open(get_f('target_albedo')))
    plt.axis('off')
    plt.title("albedo")

    plt.subplot(4, 3, 10)
    plt.imshow(Image.open(get_f('target_specular')))
    plt.axis('off')
    plt.title("specular")

    plt.subplot(4, 3, 2)
    plt.imshow(Image.open(get_f('projection')))
    plt.axis('off')
    plt.title("Opt. Inversion")

    plt.subplot(4, 3, 5)
    plt.imshow(Image.open(get_f('final_shading')))
    plt.axis('off')
    plt.title("shading")

    plt.subplot(4, 3, 8)
    plt.imshow(Image.open(get_f('final_albedo')))
    plt.axis('off')
    plt.title("albedo")

    plt.subplot(4, 3, 11)
    plt.imshow(Image.open(get_f('final_specular')))
    plt.axis('off')
    plt.title("specular")

    # plt.subplot(4,3,2)
    # plt.imshow(Image.open(f[5]))
    # plt.axis('off')
    # plt.title("Init")

    plt.subplot(4, 3, 6)
    plt.imshow(Image.open(get_f('shading_init')))
    plt.axis('off')
    plt.title("shading init")

    plt.subplot(4, 3, 9)
    plt.imshow(Image.open(get_f('albedo_init')))
    plt.axis('off')
    plt.title("albedo init")

    plt.subplot(4, 3, 12)
    plt.imshow(Image.open(get_f('specular_init')))
    plt.axis('off')
    plt.title("specular init")

    plt.show()


# %%
def plot_top_images(dirname, metric, k, kst=0):
    def get_sorted_id(dirname, metric, mode):
        m = torch.load(dirname + f'{metric}_metric.pt')
        newd = {}
        for k in m.keys():
            newd[k] = m[k][mode]
        # min(newd, key=newd.get)
        sorted_need = dict(sorted(newd.items(), key=lambda item: item[1]))
        return list(sorted_need.keys())

    top_nat = get_sorted_id(dirname, metric, 'image')
    top_al = get_sorted_id(dirname, metric, 'albedo')
    top_sh = get_sorted_id(dirname, metric, 'shading')
    top_sp = get_sorted_id(dirname, metric, 'specular')

    if metric in ['psnr', 'ssim']:
        best_ids = list(set(top_nat[-k:]).intersection(top_al[-k:], top_sh[-k:], top_sp[-k:]))
    else:
        best_ids = list(set(top_nat[:k]).intersection(top_al[:k], top_sh[:k], top_sp[:k]))

    print(best_ids)
    print(f"total {len(best_ids)} images found!")

    for i in best_ids[kst: kst + 10]:
        plot_results(i, dirname)


patterns = ['target', 'target_albedo', 'target_shading', 'target_specular', 'projection', 'final_shading',
            'final_albedo', 'final_specular', 'albedo_init', 'shading_init', 'specular_init']


def add_to_latex(source_dirname, imgids, dest_dirname, patterns, spec=True):
    if not spec:
        patterns = [v for v in patterns if 'spec' not in v]
    exptname = os.path.basename(os.path.dirname(source_dirname))
    print("Expt dir: ", exptname)
    if not os.path.exists(dest_dirname + '/' + exptname):
        os.makedirs(dest_dirname + exptname, exist_ok=True)

    def get_f(pat, img_id):
        if 'target' in pat:
            # print(sorted(glob.glob(f"{source_dirname}/{pat}_{img_id}*"))[0])
            return sorted(glob.glob(f"{source_dirname}/{pat}_{img_id}*"))[0]
        else:
            # print(sorted(glob.glob(f"{source_dirname}/{img_id}_*{pat}*"))[0])
            return sorted(glob.glob(f"{source_dirname}/{img_id}_*{pat}*"))[0]

    # copy the images to the destination folder
    for i in imgids:
        for pat in patterns:
            f = get_f(pat, i)
            os.system(f"cp {f} {dest_dirname + '/' + exptname}")

    print("Images copied!")

    latex_string = ''
    # create latex
    for j in imgids:
        latex_string += '\includegraphics[width=\s\linewidth]{{./images/{0}/{1}/{2}}}& \n \includegraphics[width=\s\linewidth]{{./images/{0}/{3}/{4}}}'.format(
            os.path.basename(os.path.dirname(dest_dirname)), exptname, os.path.basename(get_f("target", j)), exptname,
            os.path.basename(get_f("projection", j)))
        if j != imgids[-1]:
            latex_string += '& \n'
        else:
            latex_string += f'{chr(92)}{chr(92)} {chr(92)}addlinespace \n'

    for j in imgids:
        latex_string += '\includegraphics[width=\s\linewidth]{{./images/{0}/{1}/{2}}}& \n \includegraphics[width=\s\linewidth]{{./images/{0}/{3}/{4}}}'.format(
            os.path.basename(os.path.dirname(dest_dirname)), exptname, os.path.basename(get_f("target_shading", j)),
            exptname, os.path.basename(get_f("final_shading", j)))
        if j != imgids[-1]:
            latex_string += '& \n'
        else:
            latex_string += f'{chr(92)}{chr(92)} {chr(92)}addlinespace \n'

    for j in imgids:
        latex_string += '\includegraphics[width=\s\linewidth]{{./images/{0}/{1}/{2}}}& \n \includegraphics[width=\s\linewidth]{{./images/{0}/{3}/{4}}}'.format(
            os.path.basename(os.path.dirname(dest_dirname)), exptname, os.path.basename(get_f("target_albedo", j)),
            exptname, os.path.basename(get_f("final_albedo", j)))
        if j != imgids[-1]:
            latex_string += '& \n'
        else:
            latex_string += f'{chr(92)}{chr(92)} {chr(92)}addlinespace \n'

    for j in imgids:
        if spec:
            latex_string += '\includegraphics[width=\s\linewidth]{{./images/{0}/{1}/{2}}}& \n \includegraphics[width=\s\linewidth]{{./images/{0}/{3}/{4}}}'.format(
                os.path.basename(os.path.dirname(dest_dirname)), exptname,
                os.path.basename(get_f("target_specular", j)), exptname, os.path.basename(get_f("final_specular", j)))
            if j != imgids[-1]:
                latex_string += '& \n'
            else:
                latex_string += f'{chr(92)}{chr(92)} {chr(92)}addlinespace \n'

    print(latex_string)
