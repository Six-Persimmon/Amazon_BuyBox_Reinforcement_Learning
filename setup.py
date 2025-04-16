from pathlib import Path
from setuptools import setup, find_packages

# Short helper to read the README if you have one
this_dir = Path(__file__).parent
readme = (this_dir / "README.md").read_text() if (this_dir / "README.md").exists() else ""

setup(
    name="sequential_pricing_env",
    version="0.1.0",
    description="Gymnasium environment for the Klein (2021) sequential pricing duopoly",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Your Name",
    python_requires=">=3.9",
    install_requires=[
        "gymnasium>=0.29",
        "numpy>=1.22",
    ],
    packages=find_packages(exclude=("tests", "docs", "examples")),
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)