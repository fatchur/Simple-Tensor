from setuptools import setup, find_packages
from simple_tensor import *

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='simple_tensor',  # Required
    version=version,  # Required
    description='a simple package for handling tensorflow tensor',  # Required
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='fatchur_rahman',  # Optional
    author_email='fatchur.rahman1@gmail.com', #optional
    packages=["simple_tensor", 
              "simple_tensor.transfer_learning", 
              "simple_tensor.object_detector", 
              "simple_tensor.segmentation",
              "simple_tensor.rnn_packs",
              "simple_tensor.networks",
              "simple_tensor.face_recog"],  # Required
)
