import torch

class DVAETokenizer(torch.nn.Module):
    def __init__(self):
        super(DVAETokenizer, self).__init__()

    def encode(self, image):
        raise NotImplementedError

    def decode(self, input_ids):
        raise NotImplementedError

    @staticmethod
    def from_hf(repo_id):
        raise NotImplementedError

class ImageTokenizer(DVAETokenizer):
    def __init__(self, encoder, decoder, image_dim, downscale_factor):
        super(ImageTokenizer, self).__init__(encoder, decoder)
        self.image_dim = image_dim
        self.downscale_factor = downscale_factor

    def set_image_dim(self, new_dim):
        self.image_dim = new_dim
