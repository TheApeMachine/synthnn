"""
Setup script for SynthNN - Synthetic Resonant Neural Networks
"""

from setuptools import setup, find_packages

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="synthnn",
    version="0.1.0",
    author="The Ape Machine",
    description="A framework for building neural networks based on resonance and wave physics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/theapemachine/synthnn",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Sound Synthesis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "networkx>=2.6",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "music": [
            "midiutil>=1.2",
            "pyaudio>=0.2.11",
        ],
        "optimization": [
            "scikit-optimize>=0.9",
        ],
        "performance": [
            "psutil>=5.8.0",  # For system info
        ],
        "cuda": [
            "cupy-cuda11x>=10.0.0",  # For NVIDIA GPUs
            # Note: Users should install appropriate CUDA version
        ],
        "metal": [
            "mlx>=0.0.3",  # Apple's MLX framework (if available)
            # Note: Only works on Apple Silicon Macs
        ],
        "all-backends": [
            "torch>=2.0.0",  # PyTorch with MPS/CUDA support
        ],
    },
    entry_points={
        "console_scripts": [
            "synthnn-demo=synthnn.main:main",
        ],
    },
) 