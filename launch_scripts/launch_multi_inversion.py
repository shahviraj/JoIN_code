'''
Select the list of hyperparameters you want to run, and the script will launch the jobs on the cluster.
Use Task-Spooler (tsp) to run the jobs in parallel.
Set the paths accordingly.
'''

import numpy as np
import os
import glob

NETWORK_PATH = "models/materials/00014--shading-auto2-gamma10-batch32/network-snapshot-004600.pkl"
NETWORK_PATH2 = "models/materials/00006--albedo-auto4-batch48/network-snapshot-012499.pkl"
DATADIR = "datasets/materials/train/"
OUTDIR = "outputs/projections/materials_both_rgb_clipinit_single_w/"
KNN_PATH = "outputs/materials/knn/"
gpu_id = 0

imglist = [2314, 4296, 1673,3587, 4378]
alphalist = [10.0, 100.0, 1000.0, 10000.0, 100000.0]
losslist = ['elpips', 'l2', 'focal']
lrlist = [0.1, 0.01, 0.001, 0.0001]
idwlist = [0.001, 0.0001, 0.00001]


for loss in losslist:
    for img in imglist:
        for alpha in alphalist:
            for idw in idwlist:
                for lr in lrlist:
                    os.system(f"tsp python relight_projector.py \
                    --datadir={DATADIR} \
                    --outdir={OUTDIR} \
                    --target={img} \
                    --network1={NETWORK_PATH} \
                    --network2={NETWORK_PATH2} \
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
                     --gpu-id={gpu_id}")
