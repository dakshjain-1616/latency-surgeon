#!/usr/bin/env python3
"""
LatencySurgeon - Tucker-decomposed attention for HuggingFace transformers
"""

from setuptools import setup, find_packages

setup(
    name="latency-surgeon",
    version="0.1.0",
    author="LatencySurgeon Team",
    description="Surgical replacement of attention mechanisms with Tucker-decomposed attention for faster inference",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.0.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "click>=8.0.0",
        "tqdm>=4.62.0",
    ],
    entry_points={
        "console_scripts": [
            "surgeon=latency_surgeon.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)