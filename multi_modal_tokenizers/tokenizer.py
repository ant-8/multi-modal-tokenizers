import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import PIL
import json
import dall_e
from dall_e import map_pixels, unmap_pixels, load_model
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from warnings import warn

def load_state_from_repo(repo_id):
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    with open(config_path, 'r') as file:
        config = json.load(file)
    state_dict = {}
    with safe_open(model_path, framework="pt") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    return state_dict, config

def preprocess(img, target_image_size):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)

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

class DalleTokenizer(ImageTokenizer):
    def __init__(self, encoder, decoder, image_dim=192, downscale_factor=8):
        self.encoder = encoder
        self.decoder = decoder
        super(DalleTokenizer, self).__init__(
            encoder, decoder, 
            image_dim, downscale_factor
        )

    def encode(self, image):
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
        state_dict, config = load_state_from_repo(repo_id)
        model = DalleTokenizer(
            encoder=dall_e.Encoder(),
            decoder=dall_e.Decoder(),
            image_dim=config['image_dim'],
            downscale_factor=config['downscale_factor']
        )
        model.load_state_dict(state_dict)
        return model

class MixedModalTokenizer():
    def __init__(
        self, 
        text_tokenizer,
        image_tokenizer,
        device="cpu"  # This is only for image_tokenizer
    ):
        self.text_tokenizer = text_tokenizer
        self.image_tokenizer = image_tokenizer
        self.num_tokens_per_image = (image_tokenizer.image_dim // image_tokenizer.downscale_factor) ** 2
        self.device = device
        
        self.original_vocab_size = len(text_tokenizer)
        text_tokenizer.add_tokens(["<new_image>", "<image_start>", "<image_end>"])
        self.image_placement_id = text_tokenizer.convert_tokens_to_ids("<new_image>")
        self.image_start_id = text_tokenizer.convert_tokens_to_ids("<image_start>")
        self.image_end_id = text_tokenizer.convert_tokens_to_ids("<image_end>")
        self.image_id_offset = len(text_tokenizer)

    def encode(self, text="", images=[]):
        encoded_text = self.text_tokenizer.encode(text)
        if encoded_text.count(self.image_placement_id) != len(images):
            raise ValueError("The number of <new_image> tags in the text does not match the number of images provided.")
        if len(images) == 0:
            return encoded_text
   
        encoded_images = [ [x + self.image_id_offset for x in self.image_tokenizer.encode(img).cpu().tolist()] for img in images]

        i = 0
        k = 0
        while i < len(encoded_text):
            if encoded_text[i] == self.image_placement_id:
                encoded_text = encoded_text[:i] + [self.image_start_id] + encoded_images[k] + [self.image_end_id] + encoded_text[i+1:]
                k += 1
            i += 1
        return encoded_text

    def decode(self, input_ids, suppress_warnings=False):
        images = []
        i = 0
        scanning_image = False
        buf = []
        def write_buf_to_images():
            nonlocal buf
            if len(buf) > self.num_tokens_per_image:
                if suppress_warnings is False:
                    warn(f"Image token sequence is longer than expected length ({self.num_tokens_per_image}). It will be truncated.")
                buf = buf[:self.num_tokens_per_image]
            elif len(buf) < self.num_tokens_per_image:
                if suppress_warnings is False:
                    warn(f"Image token sequence is shorter than expected length ({self.num_tokens_per_image}). It will be padded to work but the image will be incomplete.")
                buf = buf + ([self.image_id_offset] * (self.num_tokens_per_image - len(buf)))
            
            images.append(
                self.image_tokenizer.decode(torch.tensor(buf, device=self.device))
            )
            buf = []
        while i < len(input_ids):
            id = input_ids[i]
            if id == self.image_start_id:
                if not scanning_image:
                    scanning_image = True
                else:
                    warn(f"Another image start tag detected before the previous one closed. Ignoring.")
            elif id == self.image_end_id:
                scanning_image = False
                write_buf_to_images()
            elif scanning_image:
                image_id = id - self.image_id_offset
                if image_id < 0:
                    if suppress_warnings is False:
                        warn(f"Read an invalid token id ({image_id}) within an image context. Ignoring.")
                else:
                    buf.append(image_id)
            i += 1

        filtered_ids = []
        for x in input_ids:
            if x >= self.image_id_offset or x == self.image_placement_id or x == self.image_start_id or x == self.image_end_id:
                continue
            filtered_ids.append(x)

        decoded_text = self.text_tokenizer.decode(filtered_ids)
        return decoded_text, images
