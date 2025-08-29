#!/usr/bin/env python3
"""
Setup script for GTF-shPLRNN ECG Classification
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gtf-shplrnn-ecg",
    version="1.0.0",
    author="Zixiang Zhou",
    author_email="your.email@university.edu",
    description="GTF-enhanced shallow Piecewise Linear RNN for Multi-label ECG Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/GTF-shPLRNN-ECG-Experiments",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=4.0.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "gpu": [
            "nvidia-ml-py3>=7.352.0",
        ],
        "apple": [
            "tensorflow-metal>=0.5.0",
        ]
    },
    include_package_data=True,
    zip_safe=False,
)