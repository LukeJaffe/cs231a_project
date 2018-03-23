# CS231A Project
This repo contains codes for a hand action recognition project.
Author: Luke Jaffe <lukejaffe.github.io>

## Directory Structure
dev.sh starts the docker container and mounts all needed directories. SDK path should be changed to your SDK path.

### docker
Contains Dockerfile with all dependencies required to run the project (aside from the Royale SDK).

### data
Contains data used in the project. Only data captured from pico is uploaded, not theSKIG dataset. This can be downloaded from https://lshao.staff.shef.ac.uk/data/SheffieldKinectGesture.htm.

### work
Contains all CNN training, testing, and data partitioning scripts.  

### pico
Contains all code related to operating the pico monstar camera, including wrappers I wrote for running everything in Python3.
