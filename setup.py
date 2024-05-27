from setuptools import setup, find_packages

setup(
    name="multi_modal_tokenizers",
    version="0.0.2",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "Pillow",
        "dall_e",
        "huggingface_hub",
        "safetensors"
    ],
    author="Anthony Nguyen",
    description="Multi-modal tokenizers for more than just text.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/anothy1/multi-modal-tokenizers",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)