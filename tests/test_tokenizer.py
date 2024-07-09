import pytest
import torch
from transformers import AutoTokenizer
from multi_modal_tokenizers import DalleTokenizer, MixedModalTokenizer, preprocess
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
    return DalleTokenizer.from_hf("anothy1/dalle-tokenizer", kwargs={'image_dim': 128})

def test_dalle_tokenizer_encode_decode(dalle_tokenizer):
    assert dalle_tokenizer.image_dim == 128, "Incorrect dalle tokenizer image_dim"
    img_url = 'https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iKIWgaiJUtss/v2/1000x-1.jpg'
    img = download_image(img_url)
    assert dalle_tokenizer.device == "cpu"
    # Encode the image
    pixel_map = preprocess(img, 128)
    tokens = dalle_tokenizer.encode_pixels(pixel_map)
    assert isinstance(tokens, torch.Tensor), "Encoded tokens should be a torch.Tensor"

    # Decode the tokens back to an image
    reconstructed = dalle_tokenizer.decode(tokens)
    assert isinstance(reconstructed, Image.Image), "Reconstructed image should be a PIL Image"
    assert len(dalle_tokenizer) == 8192, "__len__ Mismatch"

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

    for i in range(dalle_tokenizer.image_dim // dalle_tokenizer.downscale_factor):
        assert encoded.count(text_tokenizer.convert_tokens_to_ids(f"<row_{i}_start>")) == 0
        assert encoded.count(text_tokenizer.convert_tokens_to_ids(f"<row_{i}_end>")) == 0

    # Decode the sequence back to text and image
    decoded_text, decoded_images = mixed_tokenizer.decode(encoded)
    assert isinstance(decoded_text, str), "Decoded text should be a string"
    assert all(isinstance(img, Image.Image) for img in decoded_images), "Decoded images should be PIL Images"
    assert len(mixed_tokenizer) == len(text_tokenizer) + len(dalle_tokenizer), "__len__ Mismatch"

def test_mixed_modal_tokenizer_encode_decode_wrap(text_tokenizer, dalle_tokenizer):
    mixed_tokenizer = MixedModalTokenizer(
        text_tokenizer=text_tokenizer,
        image_tokenizer=dalle_tokenizer,
        wrap_rows=True
    )

    text = "This is an example with <image> in the middle."
    img_url = 'https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iKIWgaiJUtss/v2/1000x-1.jpg'
    image = download_image(img_url)

    # Encode the text and image
    encoded = mixed_tokenizer.encode(text=text, images=[image])
    assert isinstance(encoded, list), "Encoded result should be a list"

    for i in range(dalle_tokenizer.image_dim // dalle_tokenizer.downscale_factor):
        assert encoded.count(text_tokenizer.convert_tokens_to_ids(f"<row_{i}_start>")) == 1
        assert encoded.count(text_tokenizer.convert_tokens_to_ids(f"<row_{i}_end>")) == 1

    # Decode the sequence back to text and image
    decoded_text, decoded_images = mixed_tokenizer.decode(encoded)
    assert isinstance(decoded_text, str), "Decoded text should be a string"
    assert all(isinstance(img, Image.Image) for img in decoded_images), "Decoded images should be PIL Images"
    assert len(mixed_tokenizer) == len(text_tokenizer) + len(dalle_tokenizer), "__len__ Mismatch"