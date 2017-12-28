import numpy as np
import util

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
