# File: c:\Users\Vivobook\Documents\mm_detect\setup.py

"""Setup configuration for Multimodal Misinformation Detection System"""

from setuptools import setup, find_packages

setup(
    name="multimodal-misinfo",
    version="0.1.0",
    description="Multimodal misinformation detection for Vietnamese Facebook advertisements",
    long_description="""
A production-grade PyTorch system for detecting misleading Vietnamese Facebook ads
using text (PhoBERT), images (ViT-B/16), and behavioral metadata with advanced fusion.
""",
    long_description_content_type="text/markdown",
    author="Research Team",
    author_email="contact@example.com",
    url="https://github.com/example/multimodal-misinfo",
    project_urls={
        "Documentation": "https://github.com/example/multimodal-misinfo/docs",
        "Source Code": "https://github.com/example/multimodal-misinfo",
        "Bug Reports": "https://github.com/example/multimodal-misinfo/issues",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        # Core deep learning
        "torch==2.1.0",
        "torchvision==0.16.0",
        "torchaudio==2.1.0",
        # NLP & transformers
        "transformers==4.33.2",
        "timm==0.9.7",
        # Data processing
        "pandas==2.0.3",
        "numpy==1.24.3",
        "scipy==1.11.2",
        "scikit-learn==1.3.0",
        # Image processing
        "Pillow==10.0.0",
        # Configuration
        "pyyaml==6.0",
        "omegaconf==2.3.1",
        "python-dotenv==1.0.0",
        # Visualization
        "matplotlib==3.8.0",
        "seaborn==0.13.0",
        # Utilities
        "tqdm==4.66.1",
        # Monitoring
        "tensorboard==2.13.0",
        "wandb==0.15.11",
    ],
    extras_require={
        "dev": [
            "black==23.9.1",
            "isort==5.12.0",
            "flake8==6.1.0",
            "pytest==7.4.2",
            "pytest-cov==4.1.0",
        ],
        "notebooks": [
            "jupyter==1.0.0",
            "jupyterlab==4.0.6",
            "ipykernel==6.25.2",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "multimodal learning",
        "misinformation detection",
        "deep learning",
        "pytorch",
        "nlp",
        "computer vision",
        "fusion",
    ],
    include_package_data=True,
    zip_safe=False,
)
