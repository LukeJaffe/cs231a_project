#!/usr/bin/env python3

import threading
import time
import logging
import random
import queue
import numpy as np

import zmq

import cam

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

BUF_SIZE = 10
q = queue.Queue(BUF_SIZE)

class ProducerThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(ProducerThread,self).__init__()
        self.target = target
        self.name = name

    def run(self):
        cam.start_camera()

class ConsumerThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(ConsumerThread,self).__init__()
        self.target = target
        self.name = name
        return

    def run(self):
        context = zmq.Context()
        print("Connecting to hello world server...")
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:5555")
        while True:
            print('loop...')
            socket.send_string("Hello")

            # Get the reply.
            message = socket.recv()
            print("Received reply [", message, "]")
            path = message.decode("utf-8") 
            frame = np.load(path)
            print(frame.shape)
            gray_img = frame[0, :, :]
            plt.imshow(gray_img, cmap='gray')
            plt.pause(0.0001)
            plt.clf()

if __name__ == '__main__':
    
    p = ProducerThread(name='producer')
    c = ConsumerThread(name='consumer')

    p.start()
    c.start()
