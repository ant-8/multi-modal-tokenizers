

# Multi-Modal Tokenizers

Multi-modal tokenizers for more than just text. This package provides tools for tokenizing and decoding images and mixed-modal inputs (text and images) using encoders like [DALL-E's VAE](https://github.com/openai/DALL-E).

## Installation

```sh
pip install git+https://github.com/anothy1/multi-modal-tokenizers
```

Or from PyPI:

```sh
pip install multi-modal-tokenizers
```

## Usage

### Example: Using DalleTokenizer

Below is an example script demonstrating how to use the `DalleTokenizer` to encode and decode images.

```python
import requests
import PIL
import io
from multi_modal_tokenizers import DalleTokenizer, MixedModalTokenizer
from IPython.display import display

def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))

# Download an image
img = download_image('https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iKIWgaiJUtss/v2/1000x-1.jpg')

# Load the DalleTokenizer from Hugging Face repository
image_tokenizer = DalleTokenizer.from_hf("anothy1/dalle-tokenizer")

# Encode the image
tokens = image_tokenizer.encode(img)
print("Encoded tokens:", tokens)

# Decode the tokens back to an image
reconstructed = image_tokenizer.decode(tokens)

# Display the reconstructed image
display(reconstructed)
```

### Example: Using MixedModalTokenizer

The package also provides `MixedModalTokenizer` for tokenizing and decoding mixed-modal inputs (text and images).

```python
from transformers import AutoTokenizer
from multi_modal_tokenizers import MixedModalTokenizer
from PIL import Image

# Load a pretrained text tokenizer from Hugging Face
text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create a MixedModalTokenizer
mixed_tokenizer = MixedModalTokenizer(
    text_tokenizer=text_tokenizer,
    image_tokenizer=image_tokenizer,
    device="cpu"
)

# Example usage
text = "This is an example with <image> in the middle."
img_path = "path/to/your/image.jpg"
image = Image.open(img_path)

# Encode the text and image
encoded = mixed_tokenizer.encode(text=text, images=[image])
print("Encoded mixed-modal tokens:", encoded)

# Decode the sequence back to text and image
decoded_text, decoded_images = mixed_tokenizer.decode(encoded)
print("Decoded text:", decoded_text)
for idx, img in enumerate(decoded_images):
    img.save(f"decoded_image_{idx}.png")
```