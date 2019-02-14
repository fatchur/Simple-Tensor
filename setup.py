from setuptools import setup, find_packages

setup(
    name='simple_tensor',  # Required
    version='v0.0.1',  # Required
    description='a simple package for handling tensorflow tensor',  # Required
    long_description="a simple package for handling tensorflow tensor",  # Optional
    author='Chur Chur',  # Optional
    packages=["simple_tensor", "simple_tensor.transfer_learning", "simple_tensor.object_detector"],  # Required
)
