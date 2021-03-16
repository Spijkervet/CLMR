#!/bin/sh
conda env create -f environment.yml python=3.8 || conda env update -f environment.yml || exit 1

# mirdata
pip install git+git://github.com/spijkervet/mirdata@billboard

# free music archive
pip install git+git://github.com/mdeff/fma@rc1