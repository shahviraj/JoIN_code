'''
Code for evaluating the results of the projection experiments on quantitative metrics.
Change the paths in the OUTDIR and EXPT_PATHS dictionary to the paths of the experiments you want to evaluate.
'''

# %%
import lpips
from torchmetrics import MeanSquaredError
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
from tabulate import tabulate

OUTDIR = '/home/virajs/work/outputs/projections/'
EXPT_PATHS = {
    'base_clip': {
            'path':f'{OUTDIR}/primeshapesV2_batch_base_clip/',
            'img_format' : f'knn_w_{0.0}_id_w_{0.0}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}'
            },
    'base_clip_single_w': {
            'path':f'{OUTDIR}/primeshapesV2_batch_base_clip_single_w/',
            'img_format' : f'knn_w_{0.0}_id_w_{0.0}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}'
            },
    f'base_clip_in_domain_{idwlist[0]}': {
            'path':f'{OUTDIR}/primeshapesV2_batch_base_clip_in_domain/',
            'img_format' : f'knn_w_{0.0}_id_w_{idwlist[0]}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}'
            },
    f'base_clip_in_domain_{idwlist[1]}': {
            'path':f'{OUTDIR}/primeshapesV2_batch_base_clip_in_domain/',
            'img_format' : f'knn_w_{0.0}_id_w_{idwlist[1]}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}'
            },
    f'base_clip_knn_{knnwlist[0]}': {
            'path':f'{OUTDIR}/primeshapesV2_batch_base_clip_knn/',
            'img_format' : f'knn_w_{knnwlist[0]}_id_w_{0.0}_{alpha}_lr_{lr}_ns_{noise_indx}_loss_{loss}_norm_{norm_noise}_sh_{sh_fixed}_al_{al_fixed}_tm_{tm}'
            }
}


def open_img(path, device=torch.device('cuda')):
    return (torch.from_numpy(np.array(Image.open(path).convert('RGB'))) / 127.5 - 1.0).to(device).permute(2, 0, 1).unsqueeze(0)

def lpipsi(img1, img2, device):
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    return loss_fn_alex(img1, img2).item()

def l2(img1, img2, device):
    mse = MeanSquaredError().to(device)
    return mse(img1, img2).item()

def ssim(img1, img2, device):
    ms_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    return ms_ssim(img1, img2).item()

def psnr(img1, img2, device):
    mpsnr = PeakSignalNoiseRatio().to(device)
    return mpsnr(img1, img2).item()


met_fns = {
    'l2': l2,
    'psnr': psnr,
    'ssim': ssim,
    'lpips': lpipsi
}
# %%
all_expt = list(expt_paths.keys())
allvals = {}

for met in metriclist:
    allvals[met] = {'method': list(expt_paths.keys()), 'albedo': [], 'shading': [], 'image': []}

for expt in tqdm(all_expt):
    expt_dir = expt_paths[expt]['path']

    temp_sh = np.ones((len(imglist), len(metriclist)))
    temp_al = np.ones((len(imglist), len(metriclist)))
    temp_tar = np.ones((len(imglist), len(metriclist)))

    for i, img in (enumerate(imglist)):

        sh_tar = open_img(expt_dir + f'target_shading_{img}.png')
        al_tar = open_img(expt_dir + f'target_albedo_{img}.png')

        tar = open_img(expt_dir + f'target_{img}.png')

        rec = open_img(expt_dir + f'{img}_{expt_paths[expt]["img_format"]}_projection.png')

        sh_rec = open_img(expt_dir + f'{img}_{expt_paths[expt]["img_format"]}_final_shading.png')
        al_rec = open_img(expt_dir + f'{img}_{expt_paths[expt]["img_format"]}_final_albedo.png')

        for j, met in enumerate(metriclist):
            met_fn = met_fns[met]
            temp_sh[i, j] = met_fn(sh_tar, sh_rec)
            temp_al[i, j] = met_fn(al_tar, al_rec)
            temp_tar[i, j] = met_fn(tar, rec)
        if i % 100 == 99:
            print(f"Completed {i + 1} images for expt {expt}")
    for k, met in enumerate(metriclist):
        allvals[met]['albedo'].append(temp_al[:, k].mean())
        allvals[met]['shading'].append(temp_sh[:, k].mean())
        allvals[met]['image'].append(temp_tar[:, k].mean())

# %%
# generate in the form of latex-like table for easy copy-paste
for met in metriclist:
    print(f"Printing results for metric {met} ... ")
    print(tabulate(allvals[met], headers='keys', tablefmt='latex', floatfmt=".4f"))

# generate in the form of fancy table grid for easy visuslization
for met in metriclist:
    print(f"Printing results for {1000} images for metric {met} ... ")
    print(tabulate(allvals[met], headers='keys', tablefmt='fancy_grid'))
# %%
