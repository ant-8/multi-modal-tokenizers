import torch
from warnings import warn

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
