from setuptools import setup, find_packages

# Makes src a installable package
# To install the package:
# pip install -e .
setup(
    name="idatt2502-project",
    version="1.0",
    packages=find_packages("src"),
)