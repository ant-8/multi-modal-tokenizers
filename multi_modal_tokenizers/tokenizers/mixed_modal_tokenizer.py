import torch
from warnings import warn

def flatten_2d(lst):
    return [item for sublist in lst for item in sublist]

def split_into_chunks(lst, n):
    if n <= 0:
        raise ValueError("Chunk size must be greater than 0")
    
    return [lst[i:i + n] for i in range(0, len(lst), n)]

class MixedModalTokenizer:
    def __init__(
            self,
            text_tokenizer,
            image_tokenizer,
            image_placement_tag = "<image>",
            image_start_tag = "<image_start>",
            image_end_tag = "<image_end>",
            wrap_rows=False
        ):
        self.text_tokenizer = text_tokenizer
        self.image_tokenizer = image_tokenizer
        self.wrap_rows = wrap_rows

        # Calculate tokens per image based on the tokenizer's properties
        self.num_tokens_per_image = (image_tokenizer.image_dim // image_tokenizer.downscale_factor) ** 2
        
        # Extend the text tokenizer vocabulary to handle image tokens
        new_tokens = {
            'additional_special_tokens': [
                image_placement_tag, 
                image_start_tag, 
                image_end_tag
            ]
        }

        if self.wrap_rows:
            for i in range(self.image_tokenizer.image_dim):
                new_tokens['additional_special_tokens'] += [f"<row_{i}_start>", f"<row_{i}_end>"]

        text_tokenizer.add_special_tokens(new_tokens)
        self.image_placement_id, self.image_start_id, self.image_end_id = [
            text_tokenizer.convert_tokens_to_ids(token) for token in new_tokens['additional_special_tokens'][:3]
        ]
        self.image_id_offset = len(text_tokenizer)

        if self.wrap_rows:
            self.row_wrap_ids = [ text_tokenizer.convert_tokens_to_ids(f"<row_{i}_start>") for i in range(self.image_tokenizer.image_dim) ]
            self.row_wrap_ids += [ text_tokenizer.convert_tokens_to_ids(f"<row_{i}_end>") for i in range(self.image_tokenizer.image_dim) ]
        else:
            self.row_wrap_ids = []

    def set_image_dim(self, dim):
        warn("The set_image_dim() function is deprecated due to stability issues and will be removed. Instead, initialize a new ImageTokenizer with the correct image_dim first, followed by a new MixedModalTokenizer to ensure proper functionality.")
        self.image_tokenizer.image_dim = dim
        self.num_tokens_per_image = (self.image_tokenizer.image_dim // self.image_tokenizer.downscale_factor) ** 2

        if self.wrap_rows:
            new_tokens = { 'additional_special_tokens': [] }
            for i in range(self.image_tokenizer.image_dim):
                new_tokens['additional_special_tokens'] += [f"<row_{i}_start>", f"<row_{i}_end>"]
            self.text_tokenizer.add_special_tokens(new_tokens)
            self.image_id_offset = len(self.text_tokenizer)

    def __len__(self):
        result = len(self.text_tokenizer) + len(self.image_tokenizer)
        return result

    def encode(self, text="", images=[]):
        encoded_text = self.text_tokenizer.encode(text)
        if encoded_text.count(self.image_placement_id) != len(images):
            raise ValueError("Mismatch between <image> tags in text and provided images.")

        if not images:
            return encoded_text
        
        def _wrap_encoded(img):
            result = img
            if self.wrap_rows:
                result = split_into_chunks(result, self.image_tokenizer.image_dim // self.image_tokenizer.downscale_factor)
                for i in range(len(result)):
                    row_start_id = self.text_tokenizer.convert_tokens_to_ids(f"<row_{i}_start>")
                    row_start_end = self.text_tokenizer.convert_tokens_to_ids(f"<row_{i}_end>")
                    result[i] = [row_start_id] + result[i] + [row_start_end]
                result = flatten_2d(result)
            return result
        
        # Encode images and adjust token IDs with offset
        encoded_images = [
            [x + self.image_id_offset for x in self.image_tokenizer.encode(img).to('cpu').tolist()]
            for img in images
        ]

        # Inject image encodings into the text at specified positions
        result = []
        image_idx = 0
        for token in encoded_text:
            if token == self.image_placement_id and image_idx < len(encoded_images):
                result.extend([self.image_start_id] + _wrap_encoded(encoded_images[image_idx]) + [self.image_end_id])
                image_idx += 1
            else:
                result.append(token)

        return result

    def decode(self, input_ids, suppress_warnings=False):
        images, buf = [], []
        scanning_image = False

        def process_image_buffer():
            nonlocal buf
            if len(buf) > self.num_tokens_per_image:
                if not suppress_warnings:
                    warn(f"Image token sequence longer than expected ({self.num_tokens_per_image}). Truncating.")
                buf = buf[:self.num_tokens_per_image]
            elif len(buf) < self.num_tokens_per_image:
                if not suppress_warnings:
                    warn(f"Image token sequence shorter than expected ({self.num_tokens_per_image}). Padding.")
                buf += [self.image_id_offset] * (self.num_tokens_per_image - len(buf))
            images.append(self.image_tokenizer.decode(torch.tensor(buf)))
            buf = []

        # Process the encoded token IDs to extract text and images
        for id in input_ids:
            if id == self.image_start_id:
                if scanning_image:
                    warn("Nested <image_start> tag found. Ignoring.")
                scanning_image = True
            elif id == self.image_end_id and scanning_image:
                scanning_image = False
                process_image_buffer()
            elif scanning_image:
                image_id = id - self.image_id_offset
                if id in self.row_wrap_ids: continue
                if image_id >= 0:
                    buf.append(image_id)
                elif not suppress_warnings:
                    warn(f"Invalid token id ({image_id}) within image context. Ignoring.")
        
        # Remove image related IDs and decode text
        filtered_ids = [id for id in input_ids if id < self.image_id_offset and (id not in [self.image_placement_id, self.image_start_id, self.image_end_id]) and (id not in self.row_wrap_ids)]
        decoded_text = self.text_tokenizer.decode(filtered_ids)

        return decoded_text, images