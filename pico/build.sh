#!/bin/bash

rm -rf build
python3 setup.py install --user
python3 -c 'import cam'
