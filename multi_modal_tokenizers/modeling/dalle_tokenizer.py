import torch
import torch.nn.functional as F
import torchvision.transforms as T
from .base_tokenizers import ImageTokenizer
from dall_e import unmap_pixels, Encoder, Decoder

class DalleTokenizer(ImageTokenizer):
    def __init__(self, encoder, decoder, image_dim=192, downscale_factor=8):
        super(DalleTokenizer, self).__init__(
            image_dim, downscale_factor
        )
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, image):
        from ..image_processing import preprocess
        x = preprocess(image, self.image_dim).to(self.encoder.device)
        z_logits = self.encoder(x)
        ids = torch.argmax(z_logits, axis=1).flatten()
        return ids

    def decode(self, input_ids):
        grid_dim = self.image_dim // self.downscale_factor
        input_ids = input_ids.view((1, grid_dim, grid_dim))
        z = F.one_hot(input_ids, num_classes=self.encoder.vocab_size).permute(0, 3, 1, 2).float()
        x_stats = self.decoder(z).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        x_rec = T.ToPILImage(mode='RGB')(x_rec[0])
        return x_rec

    @staticmethod
    def from_hf(repo_id):
        from ..utils import load_state_from_repo
        state_dict, config = load_state_from_repo(repo_id)
        model = DalleTokenizer(
            encoder=Encoder(),
            decoder=Decoder(),
            image_dim=config['image_dim'],
            downscale_factor=config['downscale_factor']
        )
        model.load_state_dict(state_dict)
        return model
