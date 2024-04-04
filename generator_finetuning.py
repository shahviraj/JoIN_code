# Author: Viraj Shah, virajs@adobe.com; vjshah3@illinois.edu
'''
Code for fine-tuning the generator networks for the relighting task. Once the projections are obtained using optimization,
use this code for performing fine-tuning on the generator networks to get better results.
'''

from relight_utils import *

import sys
img = sys.argv[1]

import warnings
warnings.filterwarnings('ignore')

from argparse import Namespace
opts = Namespace(**{})

mpl.rcParams['figure.dpi'] = 220
mpl.rcParams['font.size'] = 4

parser = ArgumentParser()
parser.add_argument('--network1', dest='network_pkl_shading', help='Network pickle filename for shading generator',required=True, default='/shared/rsaas/vjshah3/adwork/models/gans/lumos-sh-gamma10-batch128-23756-fid-6.pkl')
parser.add_argument('--network2', dest='network_pkl_albedo', help='Network pickle filename for albedo generator',required=True, '/shared/rsaas/vjshah3/adwork/models/gans/lumos-al-paper256-batch128-24780-fid-5.pkl')
parser.add_argument('--network3', dest='network_pkl_specular', help='Network pickle filename for specular generator',required=True, '/shared/rsaas/vjshah3/adwork/models/gans/lumos-sp-batch128-23756-fid-8.pkl')
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


np.random.seed(opts.seed)
torch.manual_seed(opts.seed)
device = torch.device('cuda')

opts = update_opts(opts)

os.makedirs(opts.outdir, exist_ok=True)
os.makedirs(opts.outdir+'_pti', exist_ok=True)

impath = f'{opts.outdir}/{opts.imid}_knn_w_{opts.knn_w}_id_w_{opts.in_domain_w}_{opts.alpha}_lr_{opts.lr}_loss_{opts.loss_fn}_sh_{opts.sh_fixed}_al_{opts.al_fixed}_sp_{opts.use_specular}_data_{opts.dataset}'
impath_pti = f'{opts.outdir}_pti/{opts.imid}_knn_w_{opts.knn_w}_id_w_{opts.in_domain_w}_{opts.alpha}_lr_{opts.lr}_loss_{opts.loss_fn}_sh_{opts.sh_fixed}_al_{opts.al_fixed}_sp_{opts.use_specular}_data_{opts.dataset}'
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

final_target_image, final_sh_image, final_al_image, final_sp_image,  = project(
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


# Enter PTI code

def load_network_D(network_pkl):
    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        D = legacy.load_network_pkl(fp)['D'].requires_grad_(False).to(device)  # type: ignore
    return D

# Load the pivots
ws_sh = torch.tensor(np.load(f'{impath}_shading.npz')['w'],device=opts.device)
ws_al = torch.tensor(np.load(f'{impath}_albedo.npz')['w'], device=opts.device)

if opts.use_specular:
    ws_sp = torch.tensor(np.load(f'{impath}_specular.npz')['w'], device=opts.device)


# RUN PTI with D loss: following code tries to run pivotal tuning with Discriminator loss
ws_sh = ws_sh.detach().requires_grad_(False)
ws_al = ws_al.detach().requires_grad_(False)

opts.num_steps=1000


opts.loss_fn = 'elpips'
opts.sh_dwt = 0.001
opts.al_dwt = 0.0005
if opts.use_specular:
    opts.sp_dwt = 0.0005
opts.dist_wt = 1.0

opts.n_sup = 8

# Need to load the generators and Discriminators again
G_sh = load_network(opts.network_pkl_shading)
G_al = load_network(opts.network_pkl_albedo)

D_sh = load_network_D(opts.network_pkl_shading)
D_al = load_network_D(opts.network_pkl_albedo)

D_sh.eval()
D_al.eval()

G_sh.train().requires_grad_(True)
G_al.train().requires_grad_(True)


if opts.use_specular:
    G_sp = load_network(opts.network_pkl_specular)
    D_sp = load_network_D(opts.network_pkl_specular)
    G_sp.train().requires_grad_(True)
    D_sp.eval()


pti_lr = 0.001

pti_opt_params = list(G_sh.parameters()) + list(G_al.parameters())

if opts.use_specular:
    pti_opt_params = pti_opt_params + list(G_sp.parameters())

optimizer_pti = torch.optim.Adam(pti_opt_params, betas=(0.9, 0.999), lr=pti_lr)

if opts.loss_fn == 'lpips':
    lpips_loss = lpips.LPIPS(net='vgg').to(opts.device)
elif opts.loss_fn == 'elpips':
    elpips_loss = elpips.ELPIPS().to(opts.device)
        
target_images = tarim.to(opts.device)
if target_images.shape[2] > 256:
    target_images = F.interpolate(target_images, size=(256, 256), mode='area', recompute_scale_factor=True)
if opts.loss_fn == 'vgg' or opts.loss_fn == 'both':
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)


for step in range(int(opts.num_pti_steps)):
    # Learning rate schedule.
    t = step / (opts.num_steps*2)

    lr_ramp = min(1.0, (1.0 - t) / opts.lr_rampdown_length)
    lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
    lr_ramp = lr_ramp * min(1.0, t / opts.lr_rampup_length)
    lr = pti_lr * lr_ramp
    for param_group in optimizer_pti.param_groups:
        param_group['lr'] = lr
    
    dloss = 0.0
    if not opts.sh_fixed:
        if opts.single_w:
            synth_sh_images = G_sh.synthesis(ws_sh.repeat([1, G_sh.mapping.num_ws, 1]), noise_mode='const')
        else:
            synth_sh_images = G_sh.synthesis(ws_sh, noise_mode='const')
        dloss = dloss + opts.sh_dwt*(torch.nn.functional.softplus(-1.0*D_sh(synth_sh_images, c=None)).mean())
        z_rn = torch.randn([opts.n_sup,512], device=opts.device)
        sh_samples = G_sh.synthesis((0.9*ws_sh)+(0.1*G_sh.mapping(z_rn, None, truncation_psi=0.7)), noise_mode='const')
        dloss = dloss + opts.sh_dwt*(torch.nn.functional.softplus(-1.0*D_sh(sh_samples, c=None)).mean())
        
        synth_sh_images = gan_to_raw(synth_sh_images, 'shading', opts)
    else:
        # add pre/post processing steps
        synth_sh_images = G_sh

    if not opts.al_fixed:
        if opts.single_w:
            synth_al_images = G_al.synthesis(ws_al.repeat([1, G_al.mapping.num_ws, 1]), noise_mode='const')
        else:
            synth_al_images = G_al.synthesis(ws_al, noise_mode='const')
        dloss = dloss + opts.al_dwt*(torch.nn.functional.softplus(-1.0*D_al(synth_al_images, c=None)).mean())
        z_rn = torch.randn([opts.n_sup,512], device=opts.device)
        al_samples = G_al.synthesis((0.9*ws_al)+(0.1*G_al.mapping(z_rn, None, truncation_psi=0.7)), noise_mode='const')
        dloss = dloss + opts.al_dwt*(torch.nn.functional.softplus(-1.0*D_al(al_samples, c=None)).mean())
        
        
        synth_al_images = gan_to_raw(synth_al_images, 'albedo', opts)
    else:
        # add pre/post processing steps
        synth_al_images = G_al
    
    if opts.use_specular:
        if opts.single_w:
            synth_sp_images = G_sp.synthesis(ws_sp.repeat([1, G_sp.mapping.num_ws, 1]), noise_mode='const')
        else:
            synth_sp_images = G_sp.synthesis(ws_sp, noise_mode='const')
        dloss = dloss + opts.sp_dwt*torch.nn.functional.softplus(-1.0*D_sp(synth_sp_images, c=None)).mean()
        z_rn = torch.randn([opts.n_sup,512], device=opts.device)
        sp_samples = G_sp.synthesis((0.9*ws_sp)+(0.1*G_sp.mapping(z_rn, None, truncation_psi=0.7)), noise_mode='const')
        dloss = dloss + opts.sp_dwt*(torch.nn.functional.softplus(-1.0*D_sp(sp_samples, c=None)).mean())
        
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
    elif opts.loss_fn == 'both':
        synth_features = vgg16(synth_im_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum() + 0.5 * (
            (target_images - synth_images).square().mean())
    elif opts.loss_fn == 'lpips':
        rescaled_target_images = (target_images / 127.5) - 1.0
        rescaled_synth_images = (synth_im_images / 127.5) - 1.0
        dist = loss_fn_vgg(rescaled_target_images, rescaled_synth_images)
    elif opts.loss_fn == 'elpips':
        rescaled_target_images = (target_images / 127.5) - 1.0
        rescaled_synth_images = (synth_im_images / 127.5) - 1.0
        dist = elpips_loss(rescaled_target_images, rescaled_synth_images)
    else:
        raise NotImplementedError
    
    loss_pti = opts.dist_wt*dist + dloss

    # Step
    optimizer_pti.zero_grad(set_to_none=True)
    loss_pti.backward()
    optimizer_pti.step()

    if step % 50 == 0:
        print(f'step {step + 1:>4d}/{opts.num_pti_steps}: dist {dist.item():<4.4f} dloss {float(dloss.item()):<5.4f} total loss {float(loss_pti.item()):<5.4f}')

    # logprint(f'step {step+1:>4d}/{num_steps}: loss {float(loss):<5.2f}')

    # Save projected W for each optimization step.

    
if not opts.sh_fixed:
    if opts.single_w:
        synth_sh_image = G_sh.synthesis(ws_sh.repeat([1, G_sh.mapping.num_ws, 1]),
                                        noise_mode='const')
    else:
        synth_sh_image = G_sh.synthesis(ws_sh, noise_mode='const')
    
    synth_sh_raw_image = gan_to_raw(synth_sh_image, 'shading', opts)
    final_sh_image = gan_to_final(synth_sh_image, opts)
    final_sh_image.save(f'{impath_pti}_final_shading_pti.jpg')
    # plot_next_img(final_sh_pilimage, "Recovered Shading")
else:
    synth_sh_raw_image = G_sh
    final_sh_image = None

if not opts.al_fixed:
    if opts.single_w:
        synth_al_image = G_al.synthesis(ws_al.repeat([1, G_al.mapping.num_ws, 1]),
                                        noise_mode='const')
    else:
        synth_al_image = G_al.synthesis(ws_al, noise_mode='const')
    synth_al_raw_image = gan_to_raw(synth_al_image, 'albedo', opts)
    final_al_image = gan_to_final(synth_al_image, opts)
    final_al_image.save(f'{impath_pti}_final_albedo_pti.jpg')
else:
    synth_al_raw_image = G_al
    final_al_image = None

if opts.use_specular:
    if opts.single_w:
        synth_sp_image = G_sp.synthesis(ws_sp.repeat([1, G_sp.mapping.num_ws, 1]),
                                        noise_mode='const')
    else:
        synth_sp_image = G_sp.synthesis(ws_sp, noise_mode='const')
    synth_sp_raw_image = gan_to_raw(synth_sp_image, 'specular', opts)
    final_sp_image = gan_to_final(synth_sp_image, opts)
    final_sp_image.save(f'{impath_pti}_final_specular_pti.jpg')
else:
    synth_sp_raw_image = G_sp
    final_sp_image = None

_, final_target_image = run_forward_model(synth_al_raw_image, synth_sh_raw_image, synth_sp_raw_image, opts)
final_target_image = final_target_image.permute(0, 2, 3, 1).clamp(0, 255.0).to(torch.uint8)[0].cpu().numpy()
PIL.Image.fromarray(final_target_image, 'RGB').save(f'{impath_pti}_projection_pti.jpg')

plot_results(opts.target_fnum, opts.outdir, spec=opts.use_specular, real=True, pti=True, save=True)

if not opts.real_image:
    run_metric(final_target_image, final_sh_image, final_al_image, final_sp_image, tarim, shim, alim, spim, opts, pti=True)
else:
    run_metric(final_target_image, final_sh_image, final_al_image, final_sp_image, tarim, tarsh=None, taral=None, tarsp=None, opts=opts, pti=True)