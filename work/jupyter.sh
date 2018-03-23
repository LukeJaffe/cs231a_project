#!/bin/bash

jupyter notebook --ip=* --no-browser --allow-root \
 --NotebookApp.iopub_data_rate_limit=1000000000
