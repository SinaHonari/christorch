# File from:
# https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/LSGAN.py

import gen.generator

def get_network(z_dim):
    return gen.generator(
        input_width=28,
        input_height=28,
        output_dim=1,
        z_dim=z_dim
    )
