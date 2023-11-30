# -*- coding: utf-8 -*-
 
"""setup.py: setuptools control."""

import re
from setuptools import setup

version = re.search(
        '^__version__\s*=\s*"(.*)"',
        open('oagre/__init__.py').read(),
        re.M
    ).group(1) 

with open("README.md", "rb") as f:
    long_descr = f.read().decode("utf-8")

base_packages = [
    "scikit-learn>=0.24.1",
    "numpy",
    "scipy"
]

setup(
    name = "oagre",
    packages = ["oagre"],
    license = "MIT",
    install_requires = base_packages,
    include_package_data=True,
    version = version,
    description = "Python library for building gradient boosted meta-learner regression.",
    long_description = long_descr,
    long_description_content_type='text/markdown',
    author = "John Hawkins",
    author_email = "john@getting-data-science-done.com",
    url = "http://john-hawkins.github.io",
    project_urls = {
        'Documentation': "https://oagre.readthedocs.io",
        'Source': "https://github.com/john-hawkins/oagre",
        'Tracker': "https://github.com/john-hawkins/oagre/issues" 
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

