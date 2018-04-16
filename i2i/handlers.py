import numpy as np
from skimage.io import imsave
from skimage.transform import rescale, resize
from .. import util

def dump_visualisation(out_folder, save_every_iter=None, scale_factor=1.):
    """
    """
    def _fn(losses, inputs, outputs, kwargs):
        iter_n = kwargs['iter'] - 1
        if save_every_iter is not None:
            if iter_n % save_every_iter != 0:
                return
        else:
            if iter_n != 0:
                return
        A_real = inputs[0].data.cpu().numpy()
        B_real = inputs[1].data.cpu().numpy()
        atob, atob_btoa, btoa, btoa_atob = \
            [elem.data.cpu().numpy() for elem in outputs.values()]
        outs_np = [A_real, atob, atob_btoa, B_real, btoa, btoa_atob]
        w, h = outs_np[0].shape[-1], outs_np[0].shape[-2]
        # possible that A_real.bs != B_real.bs
        bs = np.min([outs_np[0].shape[0], outs_np[3].shape[0]])
        grid = np.zeros((h*bs, w*6, 3))
        for j in range(bs):
            for i in range(6):
                this_img = outs_np[i][j]
                this_img = outs_np[i][j]
                img_to_write = util.convert_to_rgb(this_img,
                                                   is_grayscale=False)
                grid[j*h:(j+1)*h, i*w:(i+1)*w, :] = img_to_write
        imsave(arr=rescale(grid, scale=scale_factor),
               fname="%s/%s_%i_%i.png" % (out_folder,
                                          kwargs['mode'],
                                          kwargs['epoch'],
                                          kwargs['iter']))
    return _fn
