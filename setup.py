from setuptools import setup, find_packages

setup(
    name="Deep3DFaceRecon_pytorch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "mtcnn",
        "Pillow",
        "numpy",
    ],
    include_package_data=True,
    package_data={
        # Specify package data for submodules (non-Python files)
        "Deep3DFaceRecon_pytorch.util": ["*.txt"],  # Include all .txt files in util
        "Deep3DFaceRecon_pytorch.checkpoints": ["*"],  # Include all files in checkpoints
        "Deep3DFaceRecon_pytorch.BFM": ["*"],  # Include all BFM files
    },
    description="A package for 3D face reconstruction using Deep3DFaceRecon_pytorch.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/your-github-repo",  # Optional
)
