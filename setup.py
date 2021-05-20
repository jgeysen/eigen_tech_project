#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from typing import List

from setuptools import find_packages, setup

setup_requirements = ["pytest-runner"]  # type: List[str]

test_requirements = ["pytest"]  # type: List[str]

setup(
    author="Jori Geysen",
    author_email="jorigeysen@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
    ],
    dependency_links=[],
    description="Project made during the interview process for a software engineering position at Eigen Technologies.",
    install_requires=["nltk", "pandas", "scikit-learn", "cached-property", "requests"],
    keywords="eigen_tech_project",
    name="eigen_tech_project",
    packages=find_packages(include=["eigen_tech_project"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/jgeysen/eigen_tech_project",
    version="0.1.0",
)
