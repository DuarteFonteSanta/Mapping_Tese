from setuptools import setup, find_packages

setup(
    name="neurocisuc",
    version="0.1.0",
    package_dir={"": "src"},                  # Tell Python to look in src/
    packages=find_packages(where="src"),      # Find subpackages inside src/
)