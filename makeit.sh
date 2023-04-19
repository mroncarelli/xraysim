#!/bin/bash

./clean.sh
pip install -r requirements.txt
python setup.py build_ext --inplace
python setup.py install
