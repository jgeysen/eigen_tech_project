#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import atexit
from typing import List

from setuptools import find_packages, setup
from setuptools.command.install import install

# from eigen_tech_project import nlp_models  # noqa


# class Install(_install):
#     def run(self):
#         _install.run(self)
#         import nltk  # noqa
#
#         # nltk.download()
#         nltk.download("wordnet")
#         nltk.download("averaged_perceptron_tagger")
#         nltk.download("stopwords")
#         nltk.download("punkt")


def _post_install():
    import nltk  # noqa

    nltk.download("wordnet")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("stopwords")
    nltk.download("punkt")


class NewInstall(install):
    def __init__(self, *args, **kwargs):
        super(NewInstall, self).__init__(*args, **kwargs)
        atexit.register(_post_install)


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
    install_requires=["nltk", "pandas", "scikit-learn", "cached-property"],
    keywords="eigen_tech_project",
    name="eigen_tech_project",
    packages=find_packages(include=["eigen_tech_project"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/jgeysen/eigen_tech_project",
    version="0.1.0",
    cmdclass={"install": NewInstall},
)
