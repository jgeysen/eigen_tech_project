#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

setup(
    author="Jori Geysen",
    author_email="jorigeysen@gmail.com",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
    ],
    dependency_links=[],
    description="NLP project which leverages a.o. NLTK and Sklearn to create an (in memory) inverted "
    "index for a collection of .txt-files containing textual data.",
    install_requires=["nltk", "pandas", "scikit-learn", "cached-property", "requests"],
    name="eigen_tech_project",
    packages=find_packages(include=["eigen_tech_project"]),
    url="https://github.com/jgeysen/eigen_tech_project",
    version="0.1.0",
)
