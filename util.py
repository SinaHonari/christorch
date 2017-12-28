import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


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

class ClassifierIterator():
    """
    H5 friendly iterator.
    Constructs slices [0..bs], [bs, bs*2], [bs*2, bs*3], ... and shuffles
    these at each epoch. That way the elements of the minibatch are still
    ordered w.r.t. to the entire h5 file.
    Features:
    - Allows one to convert the dataset into a binary classification
    - Deal with channels_last or channels_first inputs
    - Use in conjunction with Keras image data generator
    - Allows random crops
    """
    def __init__(self, X_arr, y_arr, bs, rnd_state=None, imgen=None, crop=None, mode='old', binary_idxs=None, data_format='channels_first'):
        """
        X_arr: a 4D h5 tensor
        y_arr: a 1d h5 list
        bs: batch size
        rnd_state:
        imgen: a Keras image data generator
        crop: height/width (in px) of random crops
        mode: TODO
        binary_idxs: only consider two classes (if there are > 2) and convert these to binary (0,1).
        """
        assert mode in ['old', 'new']
        assert data_format in ['channels_first', 'channels_last']
        self.data_format = data_format
        self.X_arr, self.y_arr, self.bs, self.rnd_state, self.imgen, self.crop, self.mode = \
            X_arr, y_arr, bs, rnd_state, imgen, crop, mode
        if binary_idxs == None:
            self.slices = _get_slices(X_arr.shape[0], bs)
            self.selected_idxs = None
            self.N = X_arr.shape[0]
        else:
            assert len(binary_idxs) == 2
            self.selected_idxs = np.where( np.in1d(y_arr[:], binary_idxs) )[0].tolist()
            self.slices = _get_slices( len(self.selected_idxs), bs)
            self.binary_idxs = binary_idxs
            self.N = len(self.selected_idxs)
        self.fn = self._fn()
    def _fn(self):
        while True:
            if self.rnd_state != None:
                self.rnd_state.shuffle(self.slices)
            for elem in self.slices:
                if self.selected_idxs == None:
                    this_X, this_y = self.X_arr[elem], self.y_arr[elem]
                else:
                    this_X, this_y = self.X_arr[ self.selected_idxs[elem] ], self.y_arr[ self.selected_idxs[elem] ]
                    this_y[ this_y == self.binary_idxs[0] ] = 0
                    this_y[ this_y == self.binary_idxs[1] ] = 1
                if self.mode == 'old':
                    # why does this work and the new version doesn't??
                    images_for_this_X = [ self._augment_image(img, self.imgen, self.crop) for img in this_X ]
                    images_for_this_X = np.asarray(images_for_this_X, dtype="float32")
                else:
                    # TODO: why does this refactored version not work????
                    # if i use this version and run my experiment, the qwk never goes >0.001...
                    assert self.crop == None
                    seed = self.rnd_state.randint(0, 100000)
                    # this imgen flow thing???
                    images_for_this_X = self.imgen.flow(this_X, None, batch_size=self.bs, seed=seed, shuffle=False).next()
                if self.data_format == 'channels_first':
                    yield images_for_this_X, this_y
                else:
                    yield images_for_this_X.swapaxes(3,2).swapaxes(2,1), this_y
    def __iter__(self):
        return self
    def next(self):
        return self.fn.next()
    def _augment_image(self, img, imgen=None, crop=None):
        img_size = img.shape[-1]
        aug_x = img
        if imgen != None:
            aug_x = imgen.flow( np.asarray([aug_x], dtype=img.dtype), None).next()[0]
        if crop != None:
            x_start = np.random.randint(0, img_size-crop+1)
            y_start = np.random.randint(0, img_size-crop+1)
            aug_x = aug_x[:, y_start:y_start+crop, x_start:x_start+crop]
        return aug_x    


def test_image_folder(batch_size):
    loader = ImageFolder("/data/lisa/data/beckhamc/dr-data/train_sample")
    train_loader = DataLoader(
        loader, batch_size=batch_size, shuffle=True, num_workers=-1)
    return train_loader

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

if __name__ == '__main__':
    #scale = transforms.Scale(255)
    #loader = test_image_folder(2)
    #for data in loader:
    #    aa,bb = data

    labels = np.asarray([0,1,2,3,4])
    tmp = int_to_ord(labels, 5)
    
    
    import pdb
    pdb.set_trace()
