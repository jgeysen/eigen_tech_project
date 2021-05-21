|Github Test| |Pre-Commit|

******************************************************
Eigen_Tech_NLP_Project
******************************************************

NLP project which leverages a.o. NLTK and Sklearn to create an (in memory) inverted index for a collection of
.txt files containing textual data.

Installation:
########

There are two options here: installation from source and installation with pip.

1. Installation from source:
=====================

Clone the public repository:

.. code-block:: console

    $ git clone git://github.com/jgeysen/eigen_tech_project

cd into the repository:

.. code-block:: console

    $ cd eigen_tech_project

and run the following:

.. code-block:: console

    $ python setup.py install


2. Installation with pip:
=====================

.. code-block:: console

    $ pip install -e git+ssh://git@github.com/jgeysen/eigen_tech_project.git@main#egg=eigen_tech_project

Setup for development (with pipenv):
########

Clone the public repository:

.. code-block:: console

    $ git clone git://github.com/jgeysen/eigen_tech_project

cd into the repository:

.. code-block:: console

    $ cd eigen_tech_project

and run the following:

.. code-block:: console

    # Install dependencies
    $ pipenv install --dev

    # Setup pre-commit and pre-push hooks
    $ pipenv run init

To activate the environment, again from the root directory of the repo:

.. code-block:: console

    pipenv shell

To create and view the documentation:

.. code-block:: console

    pipenv run make_docs


.. |GitHub Test| image:: https://github.com/jgeysen/eigen_tech_project/workflows/Test/badge.svg
   :target: https://github.com/jgeysen/eigen_tech_project/actions
   :alt: github-test
.. |Pre-Commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
