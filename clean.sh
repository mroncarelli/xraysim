#!/bin/bash

rm -rf xraysim.egg-info
rm -rf build
rm -rf dist

find . -type d -exec sh -c '(cd {} && rm -f *~; rm -f *.pyc; rm -rf __pycache__)' ';'

