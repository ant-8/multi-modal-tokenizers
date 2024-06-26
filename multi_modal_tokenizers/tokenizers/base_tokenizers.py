import torch

class DVAETokenizer(torch.nn.Module):
    def __init__(self, encoder, decoder, device):
        super(DVAETokenizer, self).__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device

    def encode(self, image):
        raise NotImplementedError

    def decode(self, input_ids):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def from_hf(repo_id):
        raise NotImplementedError

class ImageTokenizer(DVAETokenizer):
    def __init__(self, encoder, decoder, image_dim, downscale_factor, device):
        super(ImageTokenizer, self).__init__(
            encoder=encoder, 
            decoder=decoder, 
            device=device
        )
        self.image_dim = image_dim
        self.downscale_factor = downscale_factor