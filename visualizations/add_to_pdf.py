'''
python visualizations/add_to_pdf.py
This script will create a pdf file with all the images in the list imglist. The images are taken from the directory imgdir.
The images are named as grid_{img}_alpha_1000.0_varying_knn_w_loss_{loss}_sh_{sh_fixed}_al_{al_fixed}.png.
The pdf file is named as varying_alphad_compare_grid_list_singlw_w_{loss}_sh_{sh_fixed}_al_{al_fixed}.pdf.
The images are arranged in the order of the list imglist.
'''

import os
from PIL import Image
from fpdf import FPDF
import glob

pdf = FPDF()
sdir = "imageFolder/"
w,h = 0,0

loss = 'elpips'
sh_fixed = False
al_fixed = False
imglist = [39, 243, 454, 945, 86, 317]#, 2314]#, 4296, 1673, 3587, 4378]
imgdir = '/home/virajs/sensei-fs-symlink/users/virajs/work/outputs/projections/primeshapesV2_multigrids_vgginit_knn_bug_fixed_full_w/'

for i, img in enumerate(imglist):
   fname = f'{imgdir}/grid_{img}_alpha_1000.0_varying_knn_w_loss_{loss}_sh_{sh_fixed}_al_{al_fixed}.png'
    if os.path.exists(fname):
        if i == 0:
            cover = Image.open(fname)
            w,h = cover.size
            pdf = FPDF(unit = "pt", format = [w,h])
        image = fname
        pdf.add_page()
        pdf.image(image,0,0,w,h)
    else:
        print("File not found:", fname)
    print("processed %d" % i)
pdf.output(f"{imgdir}/varying_alphad_compare_grid_list_singlw_w_{loss}_sh_{sh_fixed}_al_{al_fixed}.pdf", "F")

print("done")