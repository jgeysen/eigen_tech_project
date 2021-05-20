rm -rf docs/_autosummary docs/autosummary
rm -rf docs/_static
mkdir docs/_static
make -C docs clean
make -C docs html
