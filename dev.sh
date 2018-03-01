#!/bin/bash

#XSOCK=/tmp/.X11-unix
#XAUTH=/tmp/.docker.xauth-n
#xauth nlist | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

gpudocker run \
 -i cs231a:project -c /bin/bash \
 --docker_args="-it --rm \
 --name hw_`date +%F_%H-%M-%S` \
 --net=host \
 -v ${HOME}/.vimrc:/root/.vimrc \
 -v ${HOME}/.vim:/root/.vim \
 -v `pwd`/work:/home/username/work \
 -v `pwd`/data:/home/username/data \
 -e XAUTHORITY=$XAUTH \
 -e DISPLAY=$DISPLAY \
 -e QT_X11_NO_MITSHM=1 \
 -e QT_CRASH_OUTPUT=/home/username/qt_crash.log \
 -w /home/username/work \
 --privileged"
