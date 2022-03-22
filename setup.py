import codecs
import os.path

from setuptools import find_packages, setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="dock2hit",
    version=get_version("dock2hit/__init__.py"),
    description="Module for pre-training GNNs on docking scores and predicting bioactivity.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wjm41/ugi_gnn",
    packages=['dock2hit'],
    author="W. McCorkindale",
    license="MIT License",
)
