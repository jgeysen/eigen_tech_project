#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from typing import List

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    dependency_links=[],
    description="Project made during the interview process for a software engineering position at Eigen Technologies.",
    install_requires=[],
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="eigen_tech_project",
    name="eigen_tech_project",
    packages=find_packages(include=["eigen_tech_project"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/jgeysen/eigen_tech_project",
    version="0.1.0",
    zip_safe=False,
)
