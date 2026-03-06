"""Setup script for MeshXCAD."""

from setuptools import setup, find_packages

setup(
    name="meshxcad",
    version="0.1.0",
    description="Bidirectional detail transfer between 3D meshes and parametric CAD",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
    ],
    extras_require={
        "test": ["pytest"],
    },
)
