from setuptools import setup, find_packages

setup(
    name="Deep3DFaceRecon_pytorch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "scikit-image",
        "scipy",
        "pillow",
        "ipython",
        "pyyaml",  # YAML package in pip
        "matplotlib",
        "opencv-python",
        "tensorboard",
        "tensorflow",
        "kornia",
        "dominate",
        "trimesh",
    ],
    include_package_data=True,
    package_data={
        # Specify package data for submodules (non-Python files)
        "Deep3DFaceRecon_pytorch.util": ["*.txt"],  # Include all .txt files in util
        "Deep3DFaceRecon_pytorch.checkpoints": ["*"],  # Include all files in checkpoints
        "Deep3DFaceRecon_pytorch.BFM": ["*"],  # Include all BFM files
    },
    description="A package for 3D face reconstruction using Deep3DFaceRecon_pytorch.",
    author="Alexander Huang-Menders",
    author_email="alexander.huang-menders.25@dartmouth.edu",
)