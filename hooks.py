import numpy as np
from . import util

####################
# CLASSIFIER HOOKS #
####################

def get_dump_images(how_many, out_folder):
    """
    This hook returns a function which dumps an image
    plotting some images of a minibatch on the first
    epoch.
    """
    def dump_images(X_batch, y_batch, epoch):
        size = X_batch.shape[-1]
        grid = np.zeros((size, size*how_many, 3), dtype=X_batch.dtype)
        if epoch == 1:
            # only run on the first epoch
            for i in range(how_many):
                grid[:, (i*size):(i+1)*size, : ] = util.convert_to_rgb(X_batch[i])
            from skimage.io import imsave
            imsave(arr=grid, fname="%s/dump_%i.png" % (out_folder, epoch))
    return dump_images

##################
# CYCLEGAN HOOKS #
##################

from skimage.io import imsave
from skimage.transform import rescale, resize

def cg_dump_vis(out_folder, scale_factor=1.):
    """
    """
    def _fn(losses, inputs, outputs, kwargs):
        if kwargs['iter'] != 1:
            return
        A_real = inputs[0].data.cpu().numpy()
        B_real = inputs[1].data.cpu().numpy()
        atob, atob_btoa, btoa, btoa_atob = \
            [elem.data.cpu().numpy() for elem in outputs.values()]
        outs_np = [A_real, atob, atob_btoa, B_real, btoa, btoa_atob]
        # determine # of channels
        n_channels = outs_np[0].shape[1]
        w, h = outs_np[0].shape[-1], outs_np[0].shape[-2]
        # possible that A_real.bs != B_real.bs
        bs = np.min([outs_np[0].shape[0], outs_np[3].shape[0]])
        grid = np.zeros((h*bs, w*6, 3))
        for j in range(bs):
            for i in range(6):
                n_channels = outs_np[i][j].shape[0]
                img_to_write = util.convert_to_rgb(outs_np[i][j],
                                                   is_grayscale=True if n_channels==1 else False)
                grid[j*h:(j+1)*h,i*w:(i+1)*w,:] = img_to_write
        imsave(arr=rescale(grid, scale=scale_factor),
               fname="%s/%i_%s.png" % (out_folder, kwargs['epoch'], kwargs['mode']))
    return _fn

def cg_dump_vis_a6_b3(out_folder, scale_factor=1.):
    """
    These are for images where A is a 6-channel and B is a 3-channel.
    """
    def _fn(A_real, atob, atob_btoa, B_real, btoa, btoa_atob, **kwargs):
        outs_np = [A_real, atob, atob_btoa, B_real, btoa, btoa_atob]
        # determine # of channels
        n_channels = outs_np[0].shape[1]
        shp = outs_np[0].shape[-1]
        # possible that A_real.bs != B_real.bs
        bs = np.min([outs_np[0].shape[0], outs_np[3].shape[0]])
        grid = np.zeros((shp*bs, shp*(6+2), 3))
        for j in range(bs):
            grid[j*shp:(j+1)*shp,0:shp,:] = util.convert_to_rgb(outs_np[0][j][0:3], is_grayscale=True if n_channels==1 else False) # A_real 0:3
            if outs_np[0][j].shape[0] > 3:
                grid[j*shp:(j+1)*shp,1*shp:2*shp,:] = util.convert_to_rgb(outs_np[0][j][3:6], is_grayscale=True if n_channels==1 else False) # A_real 3:6
            grid[j*shp:(j+1)*shp,2*shp:3*shp,:] = util.convert_to_rgb(outs_np[1][j][0:3], is_grayscale=True if n_channels==1 else False) # atob
            grid[j*shp:(j+1)*shp,3*shp:4*shp,:] = util.convert_to_rgb(outs_np[2][j][0:3], is_grayscale=True if n_channels==1 else False) # atob_btoa 0:3
            if outs_np[2][j].shape[0] > 3:
                grid[j*shp:(j+1)*shp,4*shp:5*shp,:] = util.convert_to_rgb(outs_np[2][j][3:6], is_grayscale=True if n_channels==1 else False) # atob_btoa 3:6
            grid[j*shp:(j+1)*shp,5*shp:6*shp,:] = util.convert_to_rgb(outs_np[3][j][0:3], is_grayscale=True if n_channels==1 else False) # b_real
            grid[j*shp:(j+1)*shp,6*shp:7*shp,:] = util.convert_to_rgb(outs_np[4][j][0:3], is_grayscale=True if n_channels==1 else False) # btoa 0:3
            grid[j*shp:(j+1)*shp,7*shp:8*shp,:] = util.convert_to_rgb(outs_np[5][j][0:3], is_grayscale=True if n_channels==1 else False) # btoa_atob
        imsave(arr=rescale(grid, scale=scale_factor), fname="%s/%i_%s.png" % (out_folder, kwargs['epoch']+1, kwargs['mode']))
    return _fn
        
