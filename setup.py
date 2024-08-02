"""
This file is used to install the package. It is used to install the package in the current environment.
"""

__author__ = "Denis Gosalci"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Denis Gosalci"
__email__ = "denisgosalci@outlook.de"
__status__ = "Development"

from setuptools import setup, find_packages

setup(
    name="project",
    version="1.0.0",
    description="This is the main package of the project",
    author="Denis Gosalci",
    author_email="denis.gosalci@iis.fraunhofer.de",
    url="https://github.com/va42xuqi/master_den",
    packages=find_packages(),
)
