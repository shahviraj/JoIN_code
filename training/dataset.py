# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import cv2

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8 or (image.dtype == np.float32 and self.datatype != None)
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------

class ExrFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        datatype,               # type of the data : albedo or shading or ....
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self.datatype = datatype
        if os.path.isdir(self._path):
            self._type = 'dir'
            if self.datatype == 'albedo':
                self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files if 'diffuse_color' in fname}
            elif self.datatype == 'shading':
                self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files if 'diffuse_dir' in fname}
            elif self.datatype == 'specular':
                self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files
                                    in os.walk(self._path) for fname in files if 'glossy_dir' in fname}
            else:
                raise NotImplementedError
                
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            if self.datatype == 'albedo':
                self._all_fnames = set([v for v in self._get_zipfile().namelist() if 'diffuse_color' in v])
            elif self.datatype == 'shading':
                self._all_fnames = set([v for v in self._get_zipfile().namelist() if 'diffuse_dir' in v])
            elif self.datatype == 'specular':
                self._all_fnames = set([v for v in self._get_zipfile().namelist() if 'glossy_dir' in v])
            else:
                raise NotImplementedError
        else:
            raise IOError('Path must point to a directory or zip')
    
        # PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in ['.exr'])
        if len(self._image_fnames) == 0:
             raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)
    
    def _tm_albedo(self, img):
         return np.where(img<0.0031308,img*12.92,(1.055*np.power(img,1/2.4))-0.055)
    
    def _tm_shading(self, img, scale=4.0):
        return scale*img / (1.0 + scale*img)

    def _tm_specular(self, img, scale=4.0):
        return scale*img / (1.0 + scale*img)

    def _read_albedo(self, fname):
        im_albedo=cv2.imread(fname,-1)
        im_albedo=cv2.cvtColor(im_albedo, cv2.COLOR_BGR2RGB)
        tm_albedo = 255.0*self._tm_albedo(im_albedo) # multiplying with 255.0 so that when it will get divided by 127.5 and subtracted 1 in training loop, it would become between 0 to 1
        return tm_albedo
    
    def _read_shading(self, fname):
        fname_dir = fname
        fname_indir = fname[:-11] + 'in' + fname[-11:]
        
        im_diff_dir = cv2.imread(fname_dir, -1)
        im_diff_dir = cv2.cvtColor(im_diff_dir, cv2.COLOR_BGR2RGB)
        
        im_diff_indir = cv2.imread(fname_indir, -1)
        im_diff_indir = cv2.cvtColor(im_diff_indir, cv2.COLOR_BGR2RGB)
        
        im_shading = (im_diff_dir + im_diff_indir)
        
        tm_shading = 255.0*self._tm_shading(im_shading)  # multiplying with 255.0 so that when it will get divided by 127.5 and subtracted 1 in training loop, it would become between 0 to 1
        
        return tm_shading

    def _read_specular(self, fname):
        #print(fname)
        fname_dir = fname
        fname_indir = fname[:-11] + 'in' + fname[-11:]

        im_gloss_dir = cv2.imread(fname_dir, -1)
        im_gloss_dir = cv2.cvtColor(im_gloss_dir, cv2.COLOR_BGR2RGB)

        im_gloss_indir = cv2.imread(fname_indir, -1)
        im_gloss_indir = cv2.cvtColor(im_gloss_indir, cv2.COLOR_BGR2RGB)

        im_gloss_shading = (im_gloss_dir + im_gloss_indir)

        fname_color = fname[:-18] + 'glossy_color' + fname[-8:]
        #print(fname_color)
        im_gloss_color = cv2.imread(fname_color, -1)
        im_gloss_color = cv2.cvtColor(im_gloss_color, cv2.COLOR_BGR2RGB)

        tm_specular = 255.0*self._tm_specular(im_gloss_color*im_gloss_shading)


        # tm_shading = 255.0 * self._tm_shading(
        #     im_shading)  # multiplying with 255.0 so that when it will get divided by 127.5 and subtracted 1 in training loop, it would become between 0 to 1

        return tm_specular

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        if pyspng is not None and self._file_ext(fname) == '.png':
            with self._open_file(fname) as f:
                image = pyspng.load(f.read())
        elif self._file_ext(fname) == '.exr':
            if self.datatype == 'albedo':
                image = self._read_albedo(os.path.join(self._path, fname))
            elif self.datatype == 'shading':
                image = self._read_shading(os.path.join(self._path, fname))
            elif self.datatype == 'specular':
                image = self._read_specular(os.path.join(self._path, fname))
            else:
                raise NotImplementedError
        else:
            image = np.array(PIL.Image.open(fname).convert('RGB'))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------

