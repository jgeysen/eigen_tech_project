|Github Test| |Pre-Commit|

******************************************************
Eigen_Tech_NLP_Project
******************************************************

Project made during the interview process for a software engineering position at Eigen Technologies.

Setup
=====

Before you do anything else, run the following from the root directory of the repo:
::

  # Install dependencies
  pipenv install --dev

  # Setup pre-commit and pre-push hooks
  pipenv run init


To activate the environment, again from the root directory of the repo:
::

  pipenv shell


Credits
=======

This package was created with Cookiecutter and the `aj-cloete/pipenv-cookiecutter <https://github.com/aj-cloete/pipenv-cookiecutter>`_ project template.

.. |GitHub Test| image:: https://github.com/jgeysen/eigen_tech_project/workflows/Test/badge.svg
   :target: https://github.com/jgeysen/eigen_tech_project/actions
   :alt: github-test
.. |Pre-Commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
