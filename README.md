## JoIN; Official PyTorch implementation

![Teaser image](./docs/stylegan2-ada-teaser-1024x252.png)

**JoIN: Joint GAN Inversion for Intrinsic Image Decomposition**<br>
Viraj Shah, Svetlana Lazebnik, Julien Philip<br>
https://arxiv.org/pdf/2305.11321.pdf<br>

## Release notes
This repository contains the official PyTorch implementation of the JoIN algorithm. JoIN is a novel method for intrinsic image decomposition that leverages the power of generative adversarial networks (GANs) to jointly invert a single image into its shading and albedo components. The method is able to produce high-quality intrinsic image decompositions that are competitive with state-of-the-art methods.

## Requirements

* We recommend Linux for performance and compatibility reasons.
* NVIDIA GPU with at least 12 GB of memory. 
* 64-bit Python 3.7 and PyTorch 1.7.1. See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.
* CUDA toolkit 11.0 or later.
* Python libraries: `pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3`. 
* Docker users: use the [provided Dockerfile](./Dockerfile) to build an image with the required library dependencies. (refer to Nvidia's official StyleGAN2 code release for more details on Docker usage)

## Getting started

This code requires pre-trained albedo, shading, and specular networks for each of the three datasets (PrimeShapes, Materials, and Lumos faces).
Models can be downloaded  from the following links:

* [PrimeShapes](https://drive.google.com/file/d/1-1Q6J9Q6J9Q6J9Q6J9Q6J9Q6J9Q6J9Q6/view?usp=sharing)
* [Materials](https://drive.google.com/file/d/1-1Q6J9Q6J9Q6J9Q6J9Q6J9Q6J9Q6J9Q6/view?usp=sharing)
* [Lumos faces](https://drive.google.com/file/d/1-1Q6J9Q6J9Q6J9Q6J9Q6J9Q6J9Q6J9Q6/view?usp=sharing)

After downloading the models, extract the contents of the zip files into the `../models` directory.

For running evaluations, one may need to download the test sets for each of the three datasets. The test sets can be downloaded from the following links:

* [PrimeShapes](https://drive.google.com/file/d/1-1Q6J9Q6J9Q6J9Q6J9Q6J9Q6J9Q6J9Q6/view?usp=sharing)
* [Materials](https://drive.google.com/file/d/1-1Q6J9Q6J9Q6J9Q6J9Q6J9Q6J9Q6J9Q6/view?usp=sharing)
* [Lumos faces](https://drive.google.com/file/d/1-1Q6J9Q6J9Q6J9Q6J9Q6J9Q6J9Q6J9Q6/view?usp=sharing)
* 
## Stage 1: Run optimization-based inversion with kNN loss

Optimization-based joint inversion is the first step in our pipeline. For synthetic datasets, this step alone is enough to obtain high-quality intrinsic image decompositions. The code automatically builds the index for computing the kNN loss, which requires `faiss` library. To install `faiss`, use `pip install faiss`.

To run the optimization-based inversion with kNN loss, use the following command:

```.bash
python relight_projector.py \
    --datadir={DATADIR} \
    --outdir={OUTDIR} \
    --target={img} \
    --network1={NETWORK_PATH_SHADING} \
    --network2={NETWORK_ALBEDO} \
    --albedo-fixed=False \
    --shading-fixed=False \
    --loss-fn={loss} \
    --alpha={alpha} \
    --lr={lr} \
    --in-domain-w={idw} \
    --knn-w=0.0 \
    --save-video=False \
    --single-w=True \
    --knn-path={KNN_PATH} \
     --gpu-id={gpu_id}
```
To launch inversion on multiple images at once, and to try different hyper-parameters, one can use `launch_scripts/launch_multi_inversion.py` script. Note that it uses [Task Spooler](github.com/task-spooler-gpu) to manage the jobs.

## Stage 2: Run the generator fine-tuning refinement
While Stage 1 is enough to obtain high-quality results on synthetic datasets, for real images, we recommend running the generator fine-tuning refinement step. This step is crucial for obtaining high-quality results on real images.

To run the generator fine-tuning refinement, use the following command:

```.bash

python generator_finetuning.py \
    --datadir={DATADIR} \
    --outdir={OUTDIR} \
    --target={img} \
    --network1={NETWORK_PATH_SHADING} \
    --network2={NETWORK_ALBEDO} \
    --albedo-fixed=False \
    --shading-fixed=False \
    --lr={lr} \
    --save-video=False \
    --single-w=True \
    --gpu-id={gpu_id}
```

## Running the evaluation

To run the quantitative evaluation to reproduce the results from the paper for MSE, PSNR, LPIPS, and SSIM scores, use the following command:

```.bash

python evaluate.py \
    --datadir={DATADIR} \
    --outdir={OUTDIR}
```

## Training the networks from scratch
### Preparing datasets

We provide the ready-to-use PrimeShapes and Materials datasets for training the networks. To prepare the Lumos faces dataset, follow the instructions below.

Step 1: Download the [Lumos Face Dataset](https://github.com/NVlabs/) and extract the images.

Step 2: Run a filteration on the images to obtain aligned images to be used in training. For that, run `datasets/face_detector.py` script.

One can also prepare the PrimeShapes dataset using Python blender library (Install with `pip install bpy`). The script `datasets/gen_primeshapes_blender.py` can be used to generate the dataset.

### Training the networks

To train the networks, use the following command:

```.bash
python train_exr.py \
    --datadir={DATADIR} \
    --outdir={OUTDIR} \
    --batch-size={batch_size} \
    --cfg='auto'
```

## Visualizing the results

For visualization and preparing the figures and tables for the paper, use the scripts in `visualizations` directory:

```.bash
python plot_multi_inversion.py 
python add_to_pdf.py
```

## Citation
```
@article{join23,
  author    = {Shah, Viraj and Lazebnik, Svetlana and Philip, Julien},
  title     = {JoIN: Joint GANs Inversion for Intrinsic Image Decomposition},
  journal   = {arXiv},
  year      = {2023},
}
```


## Acknowledgements
This work was done during an internship at Adobe Research. Parts of this codebase is based on the [StyleGAN2]() repo from Nvidia.