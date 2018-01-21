import torch
import os
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

"""
Code borrowed from Pedro Costa's vess2ret repo:
https://github.com/costapt/vess2ret
"""
def convert_to_rgb(img, is_grayscale=False):
    """Given an image, make sure it has 3 channels and that it is between 0 and 1."""
    if len(img.shape) != 3:
        raise Exception("""Image must have 3 dimensions (channels x height x width). """
                        """Given {0}""".format(len(img.shape)))
    img_ch, _, _ = img.shape
    if img_ch != 3 and img_ch != 1:
        raise Exception("""Unsupported number of channels. """
                        """Must be 1 or 3, given {0}.""".format(img_ch))
    imgp = img
    if img_ch == 1:
        imgp = np.repeat(img, 3, axis=0)
    if not is_grayscale:
        imgp = imgp * 127.5 + 127.5
        imgp /= 255.
    return np.clip(imgp.transpose((1, 2, 0)), 0, 1)

def _get_slices(length, bs):
    slices = []
    b = 0
    while True:
        if b*bs >= length:
            break
        slices.append( slice(b*bs, (b+1)*bs) )
        b += 1
    return slices

def rnd_crop(img, data_format='channels_last'):
    assert data_format in ['channels_first', 'channels_last']
    from skimage.transform import resize
    if data_format == 'channels_last':
        # (h, w, f)
        h, w = img.shape[0], img.shape[1]
    else:
        # (f, h, w)
        h, w = img.shape[1], img.shape[2]
    new_h, new_w = int(0.1*h + h), int(0.1*w + w)
    # resize only works in the format (h, w, f)
    if data_format == 'channels_first':
        img = img.swapaxes(0,1).swapaxes(1,2)
    # resize
    img_upsized = resize(img, (new_h, new_w))
    # if channels first, swap back
    if data_format == 'channels_first':
        img_upsized = img_upsized.swapaxes(2,1).swapaxes(1,0)
    h_offset = np.random.randint(0, new_h-h)
    w_offset = np.random.randint(0, new_w-w)
    if data_format == 'channels_last':
        final_img = img_upsized[h_offset:h_offset+h, w_offset:w_offset+w, :]
    else:
        final_img = img_upsized[:, h_offset:h_offset+h, w_offset:w_offset+w]
    return final_img

def min_max_then_tanh(img):
    img2 = np.copy(img)
    # old technique: if image is in [0,255],
    # if grayscale then divide by 255 (putting it in [0,1]), or
    # if colour then subtract 127.5 and divide by 127.5, putting it in [0,1].
    # we do: (x - 0) / (255)
    img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
    img2 = (img2 - 0.5) / 0.5
    return img2

def min_max(img):
    img2 = np.copy(img)
    img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
    return img2

def zmuv(img):
    img2 = np.copy(img)
    print np.min(img2), np.max(img2)
    for i in range(0, img2.shape[0]):
        print np.std(img2[i,...])
        img2[i, ...] = (img2[i, ...] - np.mean(img2[i, ...])) / np.std(img2[i,...]) # zmuv
    print np.min(img2), np.max(img2)
    return img2

def swap_axes(img):
    img2 = np.copy(img)
    img2 = img2.swapaxes(3,2).swapaxes(2,1)
    return img2


def int_to_ord(labels, num_classes):
    """
    Convert integer label to ordinal label.
    """
    ords = np.ones((len(labels), num_classes-1))
    for i in range(len(labels)):
        if labels[i]==0:
            continue
        ords[i][0:labels[i]] *= 0.
    return ords

####################################################################

def test_image_folder(batch_size):
    import torchvision.transforms as transforms
    # loads images in [0,1] initially
    loader = ImageFolder(root="/data/lisa/data/beckhamc/dr-data/train_sample",
                         transform=transforms.Compose([
                             transforms.Scale(256),
                             transforms.CenterCrop(256),
                             transforms.ToTensor(),
                             transforms.Lambda(lambda img: (img-0.5)/0.5)
                         ])
    )
    train_loader = DataLoader(
        loader, batch_size=batch_size, shuffle=True, num_workers=1)
    return train_loader

import torch.utils.data.dataset as dataset

class NumpyDataset(dataset.Dataset):
    def __init__(self, X, ys, keras_imgen=None, rnd_state=np.random.RandomState(0), reorder_channels=False):
        """
        keras_preprocessor: cannot use torchvision PIL transforms,
          so just use Keras' shit here.
        reorder_channels: in the event that X is in the form (bs, h, w, f), if this flag is set to
          True, we will reshape the x batches so that they are in the form (bs, f, h, w). Note that
          if this is required, you should make sure that the Keras data augmentor knows to use
          channels_last (TF-style tensors) rather than channels_first.
        """
        self.X = X
        if type(ys) != list:
            ys = [ys]
        for y in ys:
            assert len(y) == len(X)
        self.ys = ys
        assert len(X) == len(y)
        self.N = len(X)
        self.keras_imgen = keras_imgen
        self.rnd_state = rnd_state
        self.reorder_channels = reorder_channels
    def __getitem__(self, index):
        #xx, yy = self.X[index], self.y[index]
        xx = self.X[index]
        yy = []
        for y in self.ys:
            yy.append(y[index])
        yy = np.asarray(yy)
        if self.keras_imgen != None:
            seed = self.rnd_state.randint(0, 100000)
            xx = self.keras_imgen.flow(xx[np.newaxis], None, batch_size=1, seed=seed, shuffle=False).next()[0]
        if self.reorder_channels:
            xx = xx.swapaxes(3,2).swapaxes(2,1)
        return xx, yy
    def __len__(self):
        return self.N

from PIL import Image
    
class DatasetFromFolder(Dataset):
    """
    Courtesy of:
    https://github.com/togheppi/CycleGAN/blob/master/dataset.py
    """
    def __init__(self, image_dir, subfolder='', transform=None, resize_scale=None, crop_size=None, fliplr=False):
        super(DatasetFromFolder, self).__init__()
        self.input_path = os.path.join(image_dir, subfolder)
        self.image_filenames = [x for x in sorted(os.listdir(self.input_path))]
        self.transform = transform
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.fliplr = fliplr
    def __getitem__(self, index):
        # Load Image
        img_fn = os.path.join(self.input_path, self.image_filenames[index])
        img = Image.open(img_fn).convert('RGB')
        # preprocessing
        if self.resize_scale:
            img = img.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)
        if self.crop_size:
            x = np.random.randint(0, self.resize_scale - self.crop_size + 1)
            y = np.random.randint(0, self.resize_scale - self.crop_size + 1)
            img = img.crop((x, y, x + self.crop_size, y + self.crop_size))
        if self.fliplr:
            if np.random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.image_filenames)

if __name__ == '__main__':
    #tmp = H5Dataset()
    '''
    loader = test_image_folder(1)
    for x,y in loader:
        print x,y
        import pdb
        pdb.set_trace()
    '''

    
    
