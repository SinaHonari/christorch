from . import gen
from . import disc
#import gen.generator
#import disc.discriminator

def get_network(z_dim):
    g = gen.generator(
        input_width=28,
        input_height=28,
        output_dim=1,
        z_dim=z_dim
    )
    d = disc.discriminator(
        input_width=28,
        input_height=28,
        input_dim=1,
        output_dim=1
    )
    return {
        'g': g,
        'd': d
    }
