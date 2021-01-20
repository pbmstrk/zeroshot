NAME = "zeroshot"
AUTHOR = "Paul Baumstark"
VERSION = "0.1.0"

import os
from setuptools import setup, find_packages

PATH_ROOT = os.path.dirname(__file__)


def load_requirements(path_dir=PATH_ROOT, file_name="requirements.txt"):
    with open(os.path.join(path_dir, file_name), "r") as file:
        reqs = [ln.strip() for ln in file.readlines()]
    return reqs



setup(
    name=NAME,
    author=AUTHOR,
    version=VERSION,
    packages=find_packages(),
    install_requires=load_requirements(),
)
