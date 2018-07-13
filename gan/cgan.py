import torch
from .base import GAN

class CGAN(GAN):

    def __init__(self, y_dim, *args, **kwargs):
        self.y_dim = y_dim
        super(CGAN, self).__init__(*args, **kwargs)

    def prepare_batch(self, batch):
        if len(batch) != 2:
            raise Exception("Expected batch to only contain twos elements: " +
                            "X_batch and y_batch")
        X_batch = batch[0].float()
        y_batch = batch[1].float() # assuming one-hot encoding
        if self.use_cuda:
            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()
        return [X_batch, y_batch]

    def sample(self, bs, seed=None):
        """Return a sample G(z)"""
        self._eval()
        with torch.no_grad():
            z_batch = self.sample_z(bs, seed=seed)
            y_batch = torch.eye(self.y_dim)[
                torch.randint(self.y_dim, size=(bs,)).long() ]
            if self.use_cuda:
                y_batch = y_batch.cuda()
            gz = self.g(z_batch, y_batch)
        return gz
    
    def train_on_instance(self, z, x, y, **kwargs):
        self._train()
        # Train the generator.
        self.optim['g'].zero_grad()
        fake = self.g(z, y)
        d_fake = self.d(fake, y)
        gen_loss = self.mse(d_fake, 1)
        gen_loss.backward()
        self.optim['g'].step()
        # Train the discriminator.
        self.optim['d'].zero_grad()
        d_fake = self.d(fake.detach(), y)
        d_real = self.d(x, y)
        d_loss = self.mse(d_real, 1) + self.mse(d_fake, 0)
        d_loss.backward()
        self.optim['d'].step()
        losses = {
            'g_loss': gen_loss.data.item(),
            'd_loss': d_loss.data.item()
        }
        outputs = {
            'x': x.detach(),
            'gz': fake.detach(),
        }
        return losses, outputs
