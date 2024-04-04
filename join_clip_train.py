'''
This script trains a CLIP-based index of large number of GAN-generated images to improve the initialization for GAN Inversion.
'''

import PIL.Image
import os
import time
from relight_utils import *
import torch
import torchvision
import numpy as np
import clip
import faiss
import legacy
import dnnlib
from torchvision import models, transforms
import matplotlib as mpl
from tqdm import tqdm
mpl.rcParams['figure.dpi'] = 150
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import pyplot as plt

class _FeatureExtractor(torch.nn.Module):
    def __init__(self, pretrained):
        super(_FeatureExtractor, self).__init__()
        vgg_pretrained = models.vgg16(pretrained=pretrained).features
        # Replace maxpool by average pool

        for i, l in enumerate(vgg_pretrained):
            if isinstance(l, torch.nn.MaxPool2d):
                vgg_pretrained[i] = torch.nn.AvgPool2d(
                    kernel_size=2, stride=2, padding=0
                )
        #added 30 (last value) in the list below so that one moe block of VGG gets included (so now the output is 7x7x512 instead of 14x14x512 since last maxpool/avgpool block is also included)
        self.breakpoints = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28, 30]
        self.breakpoints = [b + 1 for b in self.breakpoints]
        self.breakpoints.insert(0, 0)

        for i, b in enumerate(self.breakpoints[:-1]):
            ops = torch.nn.Sequential()
            for idx in range(b, self.breakpoints[i + 1]):
                op = vgg_pretrained[idx]
                ops.add_module(str(idx), op)
            # print(ops)
            self.add_module("group{}".format(i), ops)

        # No gradients
        for p in self.parameters():
            p.requires_grad = False

        # Torchvision's normalization: <https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101>
        self.register_buffer(
            "shift", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "scale", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        self.dropout_masks: Dict[str, torch.Tensor] = {} #{"dummy": torch.empty(0)}

    def forward(self, x):
        feats = []
        x = (x - self.shift) / self.scale
        feats.append(x)
        gen_mask = len(self.dropout_masks) == 0
        for name, m in self.named_children():
            x = m(x)
            #print('name: ', name , 'feat shape: ', x.shape)
            feats.append(x)
            if gen_mask:
                self.dropout_masks[name] = torch.bernoulli(torch.full_like(x, 0.99))
            x *= self.dropout_masks[name]
        return feats


def _l2_normalize_features(x, eps: float = 1e-16):
    nrm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + eps)
    return x / nrm

class CLIPModel(torch.nn.Module):
    def __init__(self, clip_model_name="ViT-B/32", device="cuda"):
        super(CLIPModel, self).__init__()
        self.name = clip_model_name
        CLIPModel.__check_model_name(clip_model_name)
        self.model, self.transform = clip.load(clip_model_name, jit=False)
        self.model = self.model.requires_grad_(False).to(device)
        self.device = device

    def encode_text(self, text, norm=False):
        text_features = self.model.encode_text(clip.tokenize(text).to(self.device))
        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, image, norm=False):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image_features = self.model.encode_image(image)
        if norm:
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    @staticmethod
    def __check_model_name(clip_model_name):
        assert clip_model_name in clip.available_models(), f"Supported models are {clip.available_models()}"


class RelightCLIPCache:
    def __init__(self, clip_model, shading_model, albedo_model, mode, scale, feattype='clip',n_samples=1000, device="cuda"):
        self.clip_model = clip_model
        self.shading_model = shading_model
        self.albedo_model = albedo_model
        self.index = None
        self.mode = mode
        self.scale = scale
        self.device = device
        self.feattype = feattype
        self.gpu_resource = faiss.StandardGpuResources()
        self.n_samples = n_samples
        self.z_seeds = self.get_z_seeds()

    def build(self, index_path, cache_path, feattype='clip', batch_size=192, truncation_psi=0.5, noise_mode='const'):

        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        print(f"Generating {int(self.n_samples)} latents...")
        clip_cache = []
        latent_cache = []

        with torch.no_grad():
            for i, b in tqdm(enumerate(range(int(self.n_samples) // batch_size))):
                if self.shading_model is not None:
                    z = torch.from_numpy(np.random.RandomState(i).randn(batch_size, self.shading_model.z_dim)).float().to(self.device)
                else:
                    z = torch.from_numpy(np.random.RandomState(i).randn(batch_size, self.albedo_model.z_dim)).float().to(self.device)
                #generate shading image
                if self.shading_model is not None:
                    ws_sh = self.shading_model.mapping(z, None, truncation_psi=truncation_psi)
                    sh_img = self.shading_model.synthesis(ws_sh, noise_mode=noise_mode)

                if self.albedo_model is not None:
                    # generate albedo image
                    ws_al = self.albedo_model.mapping(z, None, truncation_psi=truncation_psi)
                    al_img = self.albedo_model.synthesis(ws_al, noise_mode=noise_mode)

                # combine synth. images to get final image
                if self.shading_model is not None and self.albedo_model is not None:
                    img = forward_pass_gan(sh_img, al_img, mode=self.mode , scale=self.scale)

                elif self.shading_model is not None:
                    img = (sh_img + 1.0) / 2.0
                
                elif self.albedo_model is not None:
                    img = (al_img + 1.0) / 2.0

                if img.shape[2] != 256 and img.shape[3] != 256:
                    img = torch.nn.functional.interpolate(img, size=(256, 256), mode='area')
                img = img[:, :,
                      img.shape[2] // 2 - 112:img.shape[2] // 2 + 112,
                      img.shape[3] // 2 - 112:img.shape[3] // 2 + 112]  # center crop

                img = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275,  0.40821073],
                                                       std=[0.26862954, 0.26130258, 0.27577711])(img)
           
                
                if feattype == 'clip':
                    clip_features = self.clip_model.encode_image(img)
                    clip_features /= clip_features.norm(dim=-1, keepdim=True)  # normalize
                    print(clip_features.shape)
                    clip_cache.extend(clip_features.unsqueeze(0).cpu())
                
                elif feattype == 'vgg':
                    # Get VGG features
                    feature_extractor = _FeatureExtractor(pretrained=True).to(self.device)
                    vgg_features = feature_extractor(img)[-1]
                    vgg_features = torch.flatten(_l2_normalize_features(vgg_features),start_dim=1)
                    #print(vgg_features.shape)
                    clip_cache.extend(vgg_features.unsqueeze(0).cpu())
        
                latent_cache.extend(z)
        clip_cache = torch.cat(clip_cache, dim=0).numpy().astype('float32')
        latent_cache = torch.stack(latent_cache, dim=0).to(torch.float32)
        #from sys import getsizeof
        #print("Size of cache array in GB: ", round(getsizeof(clip_cache) / 1024 / 1024 / 1024, 2))
        print(f"Saving latent cache and {feattype} features...")
        torch.save(latent_cache, cache_path)
        np.save(index_path[:-4]+'.npy', clip_cache)
        
        self.shading_model = self.shading_model.cpu()
        self.albedo_model = self.albedo_model.cpu()
        del self.shading_model
        del self.albedo_model
        del latent_cache
        torch.cuda.empty_cache()
        
        print(f"Building Index...")
        # Create a new index
        index = faiss.index_factory(clip_cache.shape[-1], "Flat", faiss.METRIC_INNER_PRODUCT)

        if torch.cuda.is_available():
            #ngpus = faiss.get_num_gpus()
            #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            #os.environ["CUDA_VISIBLE_DEVICES"] = f""
            index = faiss.index_cpu_to_all_gpus(index)
            #index = faiss.index_cpu_to_gpu(self.gpu_resource, 0, index)

        # Add the vectors to the index
        index.add(clip_cache)
        
        print("Index built. Saving it...")
        # Write the index to disk
        self.index = index
        if torch.cuda.is_available():
            index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, index_path)
        #torch.save(latent_cache, cache_path)
        # delete clip cache
        del clip_cache
        #del latent_cache

    def topk(self, query, k):
        assert self.index is not None, "Index is None"
        assert self.cache is not None, "Cache is None"
        # TODO: make this support more than 1 query
        # TODO: e.g. list of strings, tensors
        with torch.no_grad():
            if isinstance(query, str):
                features = self.clip_model.encode_text(query)
            else:
                if self.feattype == 'clip':
                    features = self.clip_model.encode_image(query)
                    features /= features.norm(dim=-1, keepdim=True)  # normalize
                elif self.feattype == 'vgg':
                    feature_extractor = _FeatureExtractor(pretrained=True).to(self.device)
                    features = feature_extractor(query)[-1]
                    features = torch.flatten(_l2_normalize_features(features),start_dim=1)
            
            features = features.cpu().numpy().astype('float32')

        similarity_scores, indices = self.index.search(features, k)
        return similarity_scores[0], indices[0]

    def reset(self):
        self.index = None

    def load(self, index_path, cache_path, use_gpu=False):
        index = faiss.read_index(index_path)
        cache = torch.load(cache_path)
        if use_gpu:
            index = faiss.index_cpu_to_gpu(self.gpu_resource, 0, index)
            cache = cache.to(self.device)
        self.index = index
        self.cache = cache

    def get_z_seeds(self):
        if self.shading_model is not None:
            z_list = [np.random.RandomState(seed).randn(self.shading_model.z_dim) for seed in range(self.n_samples)]
        else:
            z_list = [np.random.RandomState(seed).randn(self.albedo_model.z_dim) for seed in range(self.n_samples)]
        z = torch.from_numpy(np.stack(z_list)).float().to(self.device)
        return z


def run(shading_pkl: str,
        albedo_pkl: str,
        tm_mode: str,
        tm_scale: int,
        save_dir: str,
        datadir: str,
        feattype: str = 'clip',
        query: int = None,
        clip_name: str = "ViT-B/32",
        device: str = "cuda",
        n_samples: int = 10000,
        k: int = 3,
        truncation_psi: float = 0.5
        ):
    index_name = ""

    if shading_pkl is not None:
        print(f"Loading shading StyleGAN {os.path.basename(shading_pkl)}...")
        with dnnlib.util.open_url(shading_pkl) as f:
            shading_model = legacy.load_network_pkl(f)['G_ema'].to(device)
        shadinggan_str = os.path.basename(shading_pkl).replace(".pkl", "")
        index_name = index_name + shadinggan_str + '-'
    else:
        shading_model = None

    if albedo_pkl is not None:
        print(f"Loading albedo StyleGAN {os.path.basename(albedo_pkl)}...")
        with dnnlib.util.open_url(albedo_pkl) as f:
            albedo_model = legacy.load_network_pkl(f)['G_ema'].to(device)
        albedogan_str = os.path.basename(albedo_pkl).replace(".pkl", "")
        index_name = index_name + albedogan_str + '-'
    else:
        albedo_model = None

    print(f"Loading CLIP {clip_name}...")
    clip_model = CLIPModel(clip_model_name=clip_name, device=device)

    print(f"Loading StyleCLIP Cache...")
    styleclip_cache = RelightCLIPCache(
                                     clip_model=clip_model,
                                     shading_model=shading_model,
                                     albedo_model=albedo_model,
                                     mode = tm_mode,
                                     scale = tm_scale,
                                     feattype= feattype,
                                     n_samples=n_samples,
                                     device=device)

    clip_str = clip_name.replace("/", "-")
    index_name = index_name + f"{feattype}-{clip_str}-{n_samples}-{tm_mode}-{tm_scale}-index.bin"
    cache_name = index_name[:-4] + "-cache.pt"
    
    index_path = os.path.join(save_dir, index_name)
    cache_path = os.path.join(save_dir, cache_name)

    if query is None:
        # Build and save the index. This will take a long time (~10 minutes for 10k, hours for 1M)
        styleclip_cache.build(index_path=index_path, cache_path=cache_path, feattype=feattype)
        print("Done!")
    else:
        plot_nn_image = False
        # Read the index from disk
        styleclip_cache.load(index_path=index_path, cache_path=cache_path, use_gpu=False)
        
        for q in range(query):
            print(f"Loading query image {q}...")
            shadingraw, shadingim = load_img(datadir, int(q), 'illum')
            albedoraw, albedoim = load_img(datadir, int(q), 'reflect')
            
            if shading_pkl is None:
                targetim_org = albedoim
            elif albedo_pkl is None:
                targetim_org = shadingim
            else:
                targetraw, targetim_org = generate_im(shadingraw, albedoraw, mode='reinhard',scale=4.0)  # targetim is in 0 to 1.0 in float32 form
            
            if targetim_org.shape[2] != 256 and targetim_org.shape[3] != 256:
                targetim_org = torch.nn.functional.interpolate(targetim_org, size=(256, 256), mode='area')
            targetim = targetim_org[:, :,
                    targetim_org.shape[2] // 2 - 112:targetim_org.shape[2] // 2 + 112,
                    targetim_org.shape[3] // 2 - 112:targetim_org.shape[3] // 2 + 112]  # center crop

            targetim = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275,  0.40821073],
                                                        std=[0.26862954, 0.26130258, 0.27577711])(targetim)

            tick = time.time()
            similarity, indices = styleclip_cache.topk(targetim.to(device), k=k)
            print(f"Latency for {k}-NN query:",  time.time() - tick)
            #similarity = torch.from_numpy(similarity).float().to(device)
            # similarity = torch.softmax(similarity / 0.01, dim=-1).unsqueeze(1).unsqueeze(2).to(device)
            zs = styleclip_cache.cache[indices]
            print(zs.shape)
            nns = []
            sh_list = []
            al_list = []
            for i in range(k):
                zi = zs[i].unsqueeze(0)
                #generate shading image
                
                if shading_model is not None:
                    ws_sh = shading_model.mapping(zi, None, truncation_psi=truncation_psi)
                    sh_list.append(ws_sh)
                    if plot_nn_image:
                        sh_img = shading_model.synthesis(ws_sh, noise_mode='const')

                if albedo_model is not None:
                    # generate albedo image
                    ws_al = albedo_model.mapping(zi, None, truncation_psi=truncation_psi)
                    al_list.append(ws_al)
                    if plot_nn_image:
                        al_img = albedo_model.synthesis(ws_al, noise_mode='const')

                ws_dict = {'shading': sh_list, 'albedo': al_list}
                ws_path = save_dir + f'/{index_name.split(".")[-3]}/{q:06d}_k_{k}.pt'
                fcheck(ws_path)
                torch.save(ws_dict, ws_path)

                # combine synth. images to get final image
                if plot_nn_image:
                    if shading_model is not None and albedo_model is not None:
                        img = forward_pass_gan(sh_img, al_img, mode=tm_mode , scale=tm_scale)

                    elif shading_model is not None:
                        img = (sh_img + 1.0) / 2.0

                    elif albedo_model is not None:
                        img = (al_img + 1.0) / 2.0

                    nns.append(img)

                    # plot the images
                    fig = plt.figure(1,(6,4))
                    grid = ImageGrid(fig, 111,
                                nrows_ncols=(((k+1)//3) + 1,3),
                                axes_pad=0.3,
                                )
                    i = 0
                    image = Image.fromarray((255.0*targetim_org).permute(0, 2, 3, 1).clamp(0, 255.0).to(torch.uint8)[0].cpu().numpy(), 'RGB')
                    grid[i].imshow(image)
                    grid[i].set_title("Target")
                    grid[i].axis('off')

                    for i in range(1,k+1):
                        image = Image.fromarray((255.0*nns[i-1]).permute(0, 2, 3, 1).clamp(0, 255.0).to(torch.uint8)[0].cpu().numpy(), 'RGB')
                        grid[i].imshow(image)
                        grid[i].set_title(f"NN: {i}")
                        grid[i].axis('off')

                    plt.suptitle(f"Nearest Neighbors using CLIP embeddings, query:{q}, k: {k}, total images: {n_samples}")
                    plt.savefig(f'{save_dir}/NN_{q}_k_{k}_{index_name.split(".")[-3]}.png', bbox_inches='tight')
                    del fig, grid, image

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="/home/virajs/work/outputs/styleclip_cache/")
    parser.add_argument('--datadir', type=str, default="/home/virajs/work/data/primshapes/dataV2/")
    parser.add_argument('--clip_name', type=str, default="ViT-B/32", choices=["ViT-B/32", "ViT-B/16"])
    parser.add_argument('--shading_pkl', type=str)
    parser.add_argument('--albedo_pkl', type=str)
    parser.add_argument('--feattype', type=str, default='clip')
    parser.add_argument('--tm_scale', type=float, default=1.0)
    parser.add_argument('--tm_mode', type=str, default='reinhard')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--query', type=int) # index of the query image in the dataset; if query is given, then the index is expected to be already saved in save_dir location.
    parser.add_argument('--k', type=int, default=3)
    args = parser.parse_args()

    run(
        shading_pkl = args.shading_pkl,
        albedo_pkl = args.albedo_pkl,
        tm_mode = args.tm_mode,
        tm_scale = args.tm_scale,
        save_dir=args.save_dir,
        datadir=args.datadir,
        n_samples=args.n_samples,
        query=args.query,
        k=args.k,
        feattype=args.feattype
    )
