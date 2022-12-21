# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="pysktb",
    version="0.5.1",
    description="Scientific Python package for solving Slater Koster tight-binding topological hamiltonian",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/santoshkumarradha/pysktb",
    author="Santosh Kumar Radha",
    author_email="srr70@case.edu",
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    package_dir={"": "pysktb"},
    packages=find_packages(where="pysktb"),
    python_requires=">=3.6, <4",
    install_requires=required,
)
