#!/usr/bin/env python3

#
# Hello World client in Python
# Connects REQ socket to tcp://localhost:5555
# Sends "Hello" to server, expects "World" back
#
# INSTALLED VIA EASY_INSTALL
import zmq
import numpy as np
import matplotlib.pyplot as plt
import cam

context = zmq.Context()

# Socket to talk to server
print("Connecting to hello world server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# Do 10 requests, waiting each time for a response
for request in range(1000):
    socket.send_string("Hello")

    # Get the reply.
    message = socket.recv()
    print("Received reply ", request, "[", message, "]")
    path = message.decode("utf-8") 
    frame = np.load(path)
    print(frame.shape)
    gray_img = frame[0, :, :]
    #plt.imshow(gray_img, cmap='gray')
    #plt.pause(0.0001)
    #plt.clf()
#plt.show()
