FROM lj:tf
MAINTAINER Luke Jaffe <lukejaffe1@gmail.com>

RUN apt-get update
RUN apt-get install -y libav-tools
RUN pip3 install --upgrade torchvision

# Install zmq
RUN apt-get install -y libzmq3-dev

# Install zmq for python3
RUN pip3 install zmq

# Install zmq for C++
RUN apt-get install -y libzmqpp-dev

# Upgrade PyTorch
RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp35-cp35m-linux_x86_64.whl 

RUN apt-get install -y zlib1g-dev

# Install cnpy
RUN apt-get install -y git
WORKDIR /opt
RUN git clone https://github.com/rogersce/cnpy.git
WORKDIR cnpy
RUN mkdir build && cd build && cmake .. && make && make install
