#!/bin/bash

./clean.sh
/usr/local/bin/python3 setup.py build_ext --inplace
/usr/local/bin/python3 setup.py install
