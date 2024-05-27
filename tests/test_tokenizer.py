import pytest
import torch
from transformers import AutoTokenizer
from multi_modal_tokenizers import DalleTokenizer, MixedModalTokenizer
from PIL import Image
import requests
import io

def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))

@pytest.fixture
def text_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")

@pytest.fixture
def dalle_tokenizer():
    return DalleTokenizer.from_hf("anothy1/dalle-tokenizer")

def test_dalle_tokenizer_encode_decode(dalle_tokenizer):
    img_url = 'https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iKIWgaiJUtss/v2/1000x-1.jpg'
    img = download_image(img_url)

    # Encode the image
    tokens = dalle_tokenizer.encode(img)
    assert isinstance(tokens, torch.Tensor), "Encoded tokens should be a torch.Tensor"

    # Decode the tokens back to an image
    reconstructed = dalle_tokenizer.decode(tokens)
    assert isinstance(reconstructed, Image.Image), "Reconstructed image should be a PIL Image"

def test_mixed_modal_tokenizer_encode_decode(text_tokenizer, dalle_tokenizer):
    mixed_tokenizer = MixedModalTokenizer(
        text_tokenizer=text_tokenizer,
        image_tokenizer=dalle_tokenizer,
    )

    text = "This is an example with <image> in the middle."
    img_url = 'https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iKIWgaiJUtss/v2/1000x-1.jpg'
    image = download_image(img_url)

    # Encode the text and image
    encoded = mixed_tokenizer.encode(text=text, images=[image])
    assert isinstance(encoded, list), "Encoded result should be a list"

    # Decode the sequence back to text and image
    decoded_text, decoded_images = mixed_tokenizer.decode(encoded)
    assert isinstance(decoded_text, str), "Decoded text should be a string"
    assert all(isinstance(img, Image.Image) for img in decoded_images), "Decoded images should be PIL Images"
