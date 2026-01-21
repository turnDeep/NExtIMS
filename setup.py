# setup.py
"""
NExtIMS v1.0 - Neural EI-MS Prediction System
Setup configuration for package installation
"""

from setuptools import setup, find_packages
import os

# Version information
VERSION = "1.0.0"
DESCRIPTION = "NExtIMS v1.0 - Neural EI-MS Prediction System"

# Read README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f
        if line.strip()
        and not line.startswith("#")
        and not line.startswith("--")  # Exclude pip options (--extra-index-url, --find-links, etc.)
    ]

setup(
    name="nextims",
    version=VERSION,
    author="turnDeep",
    author_email="",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/turnDeep/NExtIMS",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            # Training
            "nextims-train=scripts.train_gnn_minimal:main",

            # Evaluation
            "nextims-evaluate=scripts.evaluate_minimal:main",

            # Inference
            "nextims-predict=scripts.predict_single:main",
            "nextims-predict-batch=scripts.predict_batch:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords=[
        "mass spectrometry",
        "EI-MS",
        "GNN",
        "graph neural network",
        "machine learning",
        "chemistry",
        "cheminformatics",
        "spectrum prediction",
    ],
    project_urls={
        "Bug Reports": "https://github.com/turnDeep/NExtIMS/issues",
        "Source": "https://github.com/turnDeep/NExtIMS",
        "Documentation": "https://github.com/turnDeep/NExtIMS/tree/main/docs",
    },
)
