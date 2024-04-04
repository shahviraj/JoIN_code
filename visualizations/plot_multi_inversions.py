'''
Code for creating the visualization of multiple inverted components for a single image.
One can choose which parameters to vary and which to keep fixed for generating the grid of results by tweaking the `style` parameter below.
'''

import os.path

import numpy as np
from PIL import Image
import glob
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import pyplot as plt

imglist = [945, 86, 317] #[5, 39, 454, 243, 945, 2314, 4296, 1673, 3587, 4378] #[5,71978, 65778]#, 69123, 10556, 8769, 45467, 63427, 34987, 3456] # [69123, 71978] #[2, 10556]
alphalist = [1000.0]#, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
losslist = ['elpips']#, 'l2', 'focal']
lrlist = [0.1]
#01, 0.001, 0.0001]
idwlist =  [0.01, 0.001]
#knnwlist  = [1.0, 0.1, 0.01, 0.001, 0.0005, 0.0001]#,1.0, 100.0, 1000.0, 1000000.0]
#knnwlist = [1e-5]
#alpha_decaylist = [400, 300, 200, 100, 0]
#varlist = [1.0, 0.1, 0.01, 0.001]

#loc = '/home/virajs/sensei-fs-symlink/users/virajs/work/outputs/projections/primeshapesV2_both_rgb_vgginit_video_trunc_rad_50_single_w/'
#loc = '/home/virajs/sensei-fs-symlink/users/virajs/work/outputs/projections/primeshapesV2_both_rgb_vgginit_video_rand_knn_trunc_1_l1_single_w/'
#loc = '/home/virajs/sensei-fs-symlink/users/virajs/work/outputs/projections/primeshapesV2_both_rgb_vgginit_video_in_domain_single_w/'
loc = '/home/virajs/sensei-fs-symlink/users/virajs/work/outputs/projections/materials_both_rgb_clipinit_single_w/'
saveloc = '/home/virajs/sensei-fs-symlink/users/virajs/work/outputs/projections/materials_multigrids_single_w/'
#saveloc = '/home/virajs/sensei-fs-symlink/users/virajs/work/outputs/projections/primeshapesV2_multigrids_trunc_rad_50_single_w/'
#saveloc = '/home/virajs/sensei-fs-symlink/users/virajs/work/outputs/projections/primeshapesV2_multigrids_in_domain_alpha1e3_single_w/'
sh_fixed = False
al_fixed = False
style = 'idwwise'
tm = 'reinhard'

if not os.path.exists(saveloc):
    os.makedirs(saveloc, exist_ok=True)
if style == 'alphawise':
    noise_indx = 0
    lr = 0.1
    norm_noise = False
    n_col = len(alphalist) + 2
    for img in imglist:
        for loss in losslist: 
            fig = plt.figure(1,(12,6))
            grid = ImageGrid(fig, 111,
                        nrows_ncols=(3,n_col),
                        axes_pad=0.3,
                        )
            i = 1
            image = Image.open(loc + f'target_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target")
            grid[i].axis('off')
        
            i = n_col
            image = Image.open(loc + f'{img}_id_w_{idw}_0.0_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_shading_init.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Init Shading")
            grid[i].axis('off')

            i = 2*n_col 
            image = Image.open(loc + f'{img}_id_w_{idw}_0.0_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_albedo_init.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Init Albedo")
            grid[i].axis('off')

            i = n_col+1
            image = Image.open(loc + f'target_shading_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target Shading")
            grid[i].axis('off')

            i = 2*n_col + 1
            image = Image.open(loc + f'target_albedo_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target Albedo")
            grid[i].axis('off')

            i = 1
            for alpha in alphalist:

                i=i+1
                
                image = Image.open(loc + f'/{img}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_projection.png'
    ).convert('RGB')
                grid[i].imshow(image)
                grid[i].set_title(f"a={alpha}")
                grid[i].axis('off')

                if sh_fixed:
                    image = Image.open(loc + f'target_shading_{img}.png').convert('RGB')
                else:
                    image = Image.open(loc + f'{img}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_final_shading.png'
    ).convert('RGB')
                grid[i+n_col].imshow(image)
                grid[i+n_col].set_title(f"Shading")
                grid[i+n_col].axis('off')

                if al_fixed:
                    image = Image.open(loc + f'target_albedo_{img}.png').convert('RGB')
                else:
                    image = Image.open(loc + f'{img}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_final_albedo.png'
    ).convert('RGB')
                grid[i+2*n_col].imshow(image)
                grid[i+2*n_col].set_title(f"Albedo")
                grid[i+2*n_col].axis('off')


            plt.suptitle(f"VGG init,results for img {img}, loss: {loss}, RGB channels, tm: reinhard, varying alpha; \n in_domain_w: {idw}, shading fixed: {sh_fixed}, albedo fixed: {al_fixed}")
            plt.savefig(f'{saveloc}+/grid_{img}_id_w_{idw}_varying_alpha_loss_{loss}_sh_{sh_fixed}_al_{al_fixed}.png', bbox_inches='tight')
            del fig, grid, image

elif style == 'losswise':

    for img in imglist:
        for alpha in alphalist: 
            n_col = len(losslist)+1
            fig = plt.figure(1,(12,6))
            grid = ImageGrid(fig, 111,
                        nrows_ncols=(3,n_col),
                        axes_pad=0.3,
                        )
            i = 1
            image = Image.open(loc + f'target_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target")
            grid[i].axis('off')
            
            i = n_col
            image = Image.open(loc + f'target_shading_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target Shading")
            grid[i].axis('off')

            i = 2*n_col
            image = Image.open(loc + f'target_albedo_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target Albedo")
            grid[i].axis('off')

            i = 0
            for loss in losslist:
                noise_indx = 0
                lr = 0.1
                i=i+1
                norm_noise = False
                image = Image.open(loc + f'{img}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_projection.png'
    ).convert('RGB')
                grid[i].imshow(image)
                grid[i].set_title(f"loss={loss}, proj.")
                grid[i].axis('off')

                if sh_fixed:
                    image = Image.open(loc + f'target_shading_{img}.png').convert('RGB')
                else:
                    image = Image.open(loc + f'{img}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_final_shading.png'
    ).convert('RGB')
                grid[i+n_col].imshow(image)
                grid[i+n_col].set_title(f"Shading")
                grid[i+n_col].axis('off')

                if al_fixed:
                    image = Image.open(loc + f'target_albedo_{img}.png').convert('RGB')
                else:
                    image = Image.open(loc + f'{img}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_final_albedo.png'
    ).convert('RGB')
                grid[i+2*n_col].imshow(image)
                grid[i+2*n_col].set_title(f"Albedo")
                grid[i+2*n_col].axis('off')


            plt.suptitle(f"Multi-inversion results for img {img}, alpha: {alpha}, varying loss, \n in_domain_w: {idw}, shading fixed: {sh_fixed}, albedo fixed: {al_fixed}")
            plt.savefig(f'{saveloc}/grid_{img}_id_w_{idw}_varying_loss_alpha_{alpha}_sh_{sh_fixed}_al_{al_fixed}.png', bbox_inches='tight')
            del fig, grid, image

if style == 'lrwise':
    alpha = 0.0
    n_col = len(lrlist)+ 1
    for img in imglist:
        for loss in losslist: 
            fig = plt.figure(1,(12,6))
            grid = ImageGrid(fig, 111,
                        nrows_ncols=(3,n_col),
                        axes_pad=0.3,
                        )
            i = 0
            image = Image.open(loc + f'target_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target")
            grid[i].axis('off')
            
            i = n_col
            image = Image.open(loc + f'target_shading_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target Shading")
            grid[i].axis('off')

            i = 2*n_col
            image = Image.open(loc + f'target_albedo_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target Albedo")
            grid[i].axis('off')

            i = 0
            for lr in lrlist:
                noise_indx = 0
                i=i+1
                norm_noise = False
                image = Image.open(loc + f'{img}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_projection.png'
    ).convert('RGB')
                grid[i].imshow(image)
                grid[i].set_title(f"lr={lr}")
                grid[i].axis('off')

                if sh_fixed:
                    image = Image.open(loc + f'target_shading_{img}.png').convert('RGB')
                else:
                    image = Image.open(loc + f'{img}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_final_shading.png'
    ).convert('RGB')
                grid[i+n_col].imshow(image)
                grid[i+n_col].set_title(f"Shading")
                grid[i+n_col].axis('off')

                if al_fixed:
                    image = Image.open(loc + f'target_albedo_{img}.png').convert('RGB')
                else:
                    image = Image.open(loc + f'{img}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_final_albedo.png'
    ).convert('RGB')
                grid[i+2*n_col].imshow(image)
                grid[i+2*n_col].set_title(f"Albedo")
                grid[i+2*n_col].axis('off')


            plt.suptitle(f"Multi-inversion results for img {img}, loss: {loss}, varying learning rate; \n in_domain_w: {idw}, alpha={alpha}; shading fixed: {sh_fixed}, albedo fixed: {al_fixed}")
            plt.savefig(f'{saveloc}/grid_{img}_id_w_{idw}_varying_lr_alpha_{alpha}_loss_{loss}_sh_{sh_fixed}_al_{al_fixed}.png', bbox_inches='tight')
            del fig, grid, image

elif style == 'idwwise':
    noise_indx = 0
    lr = 0.1
    idw = 0.01
    norm_noise = False
    n_col = len(idwlist) + 2
    loss = 'elpips'
    knnw = 0.0
    #alpha = 100000.0
    for img in imglist:
        for alpha in alphalist:
            fig = plt.figure(1, (12, 6))
            grid = ImageGrid(fig, 111,
                             nrows_ncols=(3, n_col),
                             axes_pad=0.3,
                             )
            i = 1
            image = Image.open(loc + f'target_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target")
            grid[i].axis('off')

            i = n_col
            image = Image.open(
                loc + f'{img}_knn_w_{knnw}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_shading_init.png').convert(
                'RGB')
            grid[i].imshow(image)
            grid[i].set_title("Init Shading")
            grid[i].axis('off')

            i = 2 * n_col
            image = Image.open(
                loc + f'{img}_knn_w_{knnw}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_albedo_init.png').convert(
                'RGB')
            grid[i].imshow(image)
            grid[i].set_title("Init Albedo")
            grid[i].axis('off')

            i = n_col + 1
            image = Image.open(loc + f'target_shading_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target Shading")
            grid[i].axis('off')

            i = 2 * n_col + 1
            image = Image.open(loc + f'target_albedo_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target Albedo")
            grid[i].axis('off')

            i = 1
            for idw in idwlist:

                i = i + 1

                image = Image.open(
                    loc + f'{img}_knn_w_{knnw}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_projection.png'
                    ).convert('RGB')
                grid[i].imshow(image)
                grid[i].set_title(f"id_w={idw}")
                grid[i].axis('off')

                if sh_fixed:
                    image = Image.open(loc + f'target_shading_{img}.png').convert('RGB')
                else:
                    image = Image.open(
                        loc + f'{img}_knn_w_{knnw}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_final_shading.png'
                        ).convert('RGB')
                grid[i + n_col].imshow(image)
                grid[i + n_col].set_title(f"Shading")
                grid[i + n_col].axis('off')

                if al_fixed:
                    image = Image.open(loc + f'target_albedo_{img}.png').convert('RGB')
                else:
                    image = Image.open(
                        loc + f'{img}_knn_w_{knnw}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_final_albedo.png'
                        ).convert('RGB')
                grid[i + 2 * n_col].imshow(image)
                grid[i + 2 * n_col].set_title(f"Albedo")
                grid[i + 2 * n_col].axis('off')

            plt.suptitle(
                f"NO init,results for img {img}, loss: {loss}, RGB channels, tm: reinhard, varying knn_w; \n alpha: {alpha}, idw: {idw}, shading fixed: {sh_fixed}, albedo fixed: {al_fixed}")
            plt.savefig(f'{saveloc}/grid_{img}_alpha_{alpha}_varying_knn_w_loss_{loss}_sh_{sh_fixed}_al_{al_fixed}.png',
                        bbox_inches='tight')
            del fig, grid, image


elif style == 'knnwwise':
    noise_indx = 0
    lr = 0.1
    idw = 0.0
    norm_noise = False
    n_col = 2*len(knnwlist) + 2
    loss = 'elpips'
    #alpha = 100000.0
    for img in imglist:
        for alpha in alphalist:
            fig = plt.figure(1, (12, 6))
            grid = ImageGrid(fig, 111,
                             nrows_ncols=(3, n_col),
                             axes_pad=0.3,
                             )
            i = 1
            image = Image.open(loc + f'target_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target")
            grid[i].axis('off')

            i = n_col
            image = Image.open(
                loc + f'{img}_knn_w_{knnwlist[0]}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_shading_init.png').convert(
                'RGB')
            grid[i].imshow(image)
            grid[i].set_title("Init Shading")
            grid[i].axis('off')

            i = 2 * n_col
            image = Image.open(
                loc + f'{img}_knn_w_{knnwlist[0]}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_albedo_init.png').convert(
                'RGB')
            grid[i].imshow(image)
            grid[i].set_title("Init Albedo")
            grid[i].axis('off')

            i = n_col + 1
            image = Image.open(loc + f'target_shading_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target Shading")
            grid[i].axis('off')

            i = 2 * n_col + 1
            image = Image.open(loc + f'target_albedo_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target Albedo")
            grid[i].axis('off')

            i = 1
            for idwk in knnwlist:

                i = i + 1

                image = Image.open(
                    loc + f'{img}_knn_w_{idwk}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_projection.png'
                    ).convert('RGB')
                grid[i].imshow(image)
                grid[i].set_title(f"knn_w={idwk}")
                grid[i].axis('off')

                image = Image.open(
                    loc + f'{img}_knn_w_{idwk}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_projection_pti.png'
                ).convert('RGB')
                grid[i+1].imshow(image)
                grid[i+1].set_title(f"PTI")
                grid[i+1].axis('off')

                if sh_fixed:
                    image = Image.open(loc + f'target_shading_{img}.png').convert('RGB')
                else:
                    image = Image.open(
                        loc + f'{img}_knn_w_{idwk}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_final_shading.png'
                        ).convert('RGB')
                grid[i + n_col].imshow(image)
                grid[i + n_col].set_title(f"Shading")
                grid[i + n_col].axis('off')

                if sh_fixed:
                    image = Image.open(loc + f'target_shading_{img}.png').convert('RGB')
                else:
                    image = Image.open(
                        loc + f'{img}_knn_w_{idwk}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_final_shading_pti.png'
                        ).convert('RGB')
                grid[i + n_col + 1].imshow(image)
                grid[i + n_col + 1].set_title(f"PTI-Shading")
                grid[i + n_col + 1].axis('off')


                if al_fixed:
                    image = Image.open(loc + f'target_albedo_{img}.png').convert('RGB')
                else:
                    image = Image.open(
                        loc + f'{img}_knn_w_{idwk}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_final_albedo.png'
                        ).convert('RGB')
                grid[i + 2 * n_col].imshow(image)
                grid[i + 2 * n_col].set_title(f"Albedo")
                grid[i + 2 * n_col].axis('off')

                if al_fixed:
                    image = Image.open(loc + f'target_albedo_{img}.png').convert('RGB')
                else:
                    image = Image.open(
                        loc + f'{img}_knn_w_{idwk}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_final_albedo_pti.png'
                        ).convert('RGB')
                grid[i + 2 * n_col + 1].imshow(image)
                grid[i + 2 * n_col + 1].set_title(f"PTI-Albedo")
                grid[i + 2 * n_col + 1].axis('off')

            plt.suptitle(
                f"VGG init,results for img {img}, loss: {loss}, RGB channels, tm: reinhard, varying knn_w; \n alpha: {alpha}, idw: {idw}, shading fixed: {sh_fixed}, albedo fixed: {al_fixed}")
            plt.savefig(f'{saveloc}/grid_{img}_alpha_{alpha}_varying_knn_w_loss_{loss}_sh_{sh_fixed}_al_{al_fixed}.png',
                        bbox_inches='tight')
            del fig, grid, image


elif style == 'alphadwise':
    noise_indx = 0
    lr = 0.1
    idw = 0.0
    norm_noise = False
    n_col = len(alpha_decaylist) + 2
    loss = 'elpips'
    knnw = 1e-5
    #alpha = 100000.0
    for img in imglist:
        for alpha in alphalist:
            fig = plt.figure(1, (12, 6))
            grid = ImageGrid(fig, 111,
                             nrows_ncols=(3, n_col),
                             axes_pad=0.3,
                             )
            i = 1
            image = Image.open(loc + f'target_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target")
            grid[i].axis('off')

            i = n_col
            image = Image.open(loc + f'{img}_alphad_{alpha_decaylist[0]}_knn_w_{knnw}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_shading_init.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Init Shading")
            grid[i].axis('off')

            i = 2 * n_col
            image = Image.open(
                loc + f'{img}_alphad_{alpha_decaylist[0]}_knn_w_{knnw}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_albedo_init.png').convert(
                'RGB')
            grid[i].imshow(image)
            grid[i].set_title("Init Albedo")
            grid[i].axis('off')

            i = n_col + 1
            image = Image.open(loc + f'target_shading_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target Shading")
            grid[i].axis('off')

            i = 2 * n_col + 1
            image = Image.open(loc + f'target_albedo_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target Albedo")
            grid[i].axis('off')

            i = 1
            for alphad in alpha_decaylist:

                i = i + 1

                image = Image.open(
                    loc + f'{img}_alphad_{alphad}_knn_w_{knnw}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_projection.png'
                    ).convert('RGB')
                grid[i].imshow(image)
                grid[i].set_title(f"a_decay={alphad}")
                grid[i].axis('off')

                if sh_fixed:
                    image = Image.open(loc + f'target_shading_{img}.png').convert('RGB')
                else:
                    image = Image.open(
                        loc + f'{img}_alphad_{alphad}_knn_w_{knnw}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_final_shading.png'
                        ).convert('RGB')
                grid[i + n_col].imshow(image)
                grid[i + n_col].set_title(f"Shading")
                grid[i + n_col].axis('off')

                if al_fixed:
                    image = Image.open(loc + f'target_albedo_{img}.png').convert('RGB')
                else:
                    image = Image.open(
                        loc + f'{img}_alphad_{alphad}_knn_w_{knnw}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_final_albedo.png'
                        ).convert('RGB')
                grid[i + 2 * n_col].imshow(image)
                grid[i + 2 * n_col].set_title(f"Albedo")
                grid[i + 2 * n_col].axis('off')

            plt.suptitle(
                f"CLIP init,results for img {img}, loss: {loss}, RGB channels, tm: reinhard, varying alpha with lr; \n alpha: {alpha}, idw: {idw}, shading fixed: {sh_fixed}, albedo fixed: {al_fixed}")
            plt.savefig(f'{saveloc}/grid_{img}_alpha_{alpha}_knnw_{knnw}_varying_alphad_loss_{loss}_sh_{sh_fixed}_al_{al_fixed}.png',
                        bbox_inches='tight')
            del fig, grid, image

elif style == 'varwise':
    noise_indx = 0
    lr = 0.1
    idw = 0.0
    norm_noise = False
    n_col = len(alpha_decaylist) + 2
    loss = 'elpips'
    knnw = 1e-5
    #alpha = 100000.0
    for img in imglist:
        for alpha in alphalist:
            fig = plt.figure(1, (12, 6))
            grid = ImageGrid(fig, 111,
                             nrows_ncols=(3, n_col),
                             axes_pad=0.3,
                             )
            i = 1
            image = Image.open(loc + f'target_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target")
            grid[i].axis('off')

            i = n_col
            image = Image.open(loc + f'{img}_var_{varlist[0]}_knn_w_{knnw}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_shading_init.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Init Shading")
            grid[i].axis('off')

            i = 2 * n_col
            image = Image.open(
                loc + f'{img}_var_{varlist[0]}_knn_w_{knnw}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_albedo_init.png').convert(
                'RGB')
            grid[i].imshow(image)
            grid[i].set_title("Init Albedo")
            grid[i].axis('off')

            i = n_col + 1
            image = Image.open(loc + f'target_shading_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target Shading")
            grid[i].axis('off')

            i = 2 * n_col + 1
            image = Image.open(loc + f'target_albedo_{img}.png').convert('RGB')
            grid[i].imshow(image)
            grid[i].set_title("Target Albedo")
            grid[i].axis('off')

            i = 1
            for var in varlist:

                i = i + 1

                image = Image.open(
                    loc + f'{img}_var_{var}_knn_w_{knnw}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_projection.png'
                    ).convert('RGB')
                grid[i].imshow(image)
                grid[i].set_title(f"var={var}")
                grid[i].axis('off')

                if sh_fixed:
                    image = Image.open(loc + f'target_shading_{img}.png').convert('RGB')
                else:
                    image = Image.open(
                        loc + f'{img}_var_{var}_knn_w_{knnw}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_final_shading.png'
                        ).convert('RGB')
                grid[i + n_col].imshow(image)
                grid[i + n_col].set_title(f"Shading")
                grid[i + n_col].axis('off')

                if al_fixed:
                    image = Image.open(loc + f'target_albedo_{img}.png').convert('RGB')
                else:
                    image = Image.open(
                        loc + f'{img}_var_{var}_knn_w_{knnw}_id_w_{idw}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}_final_albedo.png'
                        ).convert('RGB')
                grid[i + 2 * n_col].imshow(image)
                grid[i + 2 * n_col].set_title(f"Albedo")
                grid[i + 2 * n_col].axis('off')

            plt.suptitle(
                f"CLIP init,results for img {img}, loss: {loss}, RGB channels, tm: reinhard, varying var; \n alpha: {alpha}, idw: {idw}, shading fixed: {sh_fixed}, albedo fixed: {al_fixed}")
            plt.savefig(f'{saveloc}/grid_{img}_alpha_{alpha}_knnw_{knnw}_varying_var_loss_{loss}_sh_{sh_fixed}_al_{al_fixed}.png',
                        bbox_inches='tight')
            del fig, grid, image