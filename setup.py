"""Setup script for VM Search."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vm_search",
    version="0.1.0",
    author="Your Name",
    description="An image searching engine using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "faiss-cpu>=1.7.4",
        "timm>=0.9.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
    ],
)
