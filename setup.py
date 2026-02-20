from setuptools import setup, find_packages

setup(
    name="mapping_tese",
    version="0.1.0",
    description="Somatosensory Mapping Analysis",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "jupyter",
        "jupyterlab",
        "nilearn",
        "nibabel",
        "pytorch"
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
)
