|Github Test| |Pre-Commit|

******************************************************
Eigen_Tech_NLP_Project
******************************************************

NLP project which leverages a.o. NLTK and Sklearn to create an (in memory) inverted index for a collection of
.txt files containing textual data.

Installation:
########

Installation from source:
=====================

The source for Eigen_Tech_NLP_Project can be downloaded from the `Github repo`_:

Clone the public repository:

.. code-block:: console

    $ git clone git://github.com/jgeysen/eigen_tech_project

Once you have a copy of the source, cd into the project:

.. code-block:: console

    $ cd eigen_tech_project

and you can install it with:

.. code-block:: console

    $ python setup.py install


Installation with pip:
=====================

.. code-block:: console

    $ pip install -e git+ssh://git@github.com/jgeysen/eigen_tech_project.git@main#egg=eigen_tech_project

Setup for development:
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


.. |GitHub Test| image:: https://github.com/jgeysen/eigen_tech_project/workflows/Test/badge.svg
   :target: https://github.com/jgeysen/eigen_tech_project/actions
   :alt: github-test
.. |Pre-Commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
