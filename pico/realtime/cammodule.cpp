/****************************************************************************\
 * Copyright (C) 2017 Infineon Technologies & pmdtechnologies ag
 *
 * THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
 * KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
 * PARTICULAR PURPOSE.
 *
 \****************************************************************************/

#include "Python.h"

#include "cnpy.h"

#include <zmq.hpp>

#include <royale.hpp>
#include <iostream>
#include <mutex>
#include <queue>

using namespace royale;
using namespace std;

// Linker errors for the OpenCV sample
//
// If this example gives linker errors about undefined references to cv::namedWindow and cv::imshow,
// or QFontEngine::glyphCache and qMessageFormatString (from OpenCV to Qt), it may be caused by a
// change in the compiler's C++ ABI.
//
// With Ubuntu and Debian's distribution packages, the libopencv packages that have 'v5' at the end
// of their name, for example libopencv-video2.4v5, are compatible with GCC 5 (and GCC 6), but
// incompatible with GCC 4.8 and GCC 4.9. The -dev packages don't have the postfix, but depend on
// the v5 (or non-v5) version of the corresponding lib package.  When Ubuntu moves to OpenCV 3.0,
// they're likely to drop the postfix (but the packages will be for GCC 5 or later).
//
// If you are manually installing OpenCV or Qt, you need to ensure that the binaries were compiled
// with the same version of the compiler.  The version number of the packages themselves doesn't say
// which ABI they use, it depends on which version of the compiler was used.

// this represents the main camera device object
std::unique_ptr<ICameraDevice> cameraDevice;

void my_handler(int s){
           printf("Caught signal %d\n",s);
            if (cameraDevice->stopCapture() != CameraStatus::SUCCESS)
            {
                cerr << "Error stopping the capturing" << endl;
            }
           exit(1); 
}


class MyListener : public IDepthDataListener
{

public :

    uint32_t m_frameNumber; // the current frame number
    std::queue<std::string>* q;

    MyListener()
    {
        m_frameNumber = 0;
    }
   
    void lock()
    {
        std::lock_guard<std::mutex> lock (flagMutex);
    }

    void onNewData (const DepthData *data)
    {
        // this callback function will be called for every new
        // depth frame

        lock();

        cout << "Receiving frame: " << m_frameNumber << endl;

        int Nx = 2;
        int Ny = 287;
        int Nz = 352;
        std::vector<float> vec(Nx*Ny*Nz);

        int k = 0;
        int mode_offset = data->width*data->height;
        for (int y = 0; y < data->height; y++)
        {
            for (int x = 0; x < data->width; x++, k++)
            {
                auto curPoint = data->points.at (k);
                if (curPoint.depthConfidence > 0)
                {
                    // if the point is valid, map the pixel from 3D world
                    // coordinates to a 2D plane (this will distort the image)
                    float z = curPoint.z;
                    float g = curPoint.grayValue;
                    // Set elements of frame vector
                    vec[k] = g;
                    vec[mode_offset+k] = z;
                }
            }
        }
        std::string path = "data/arr" + std::to_string(m_frameNumber) + ".npy";
        cnpy::npy_save(path, &vec[0], {Nx,Ny,Nz}, "w");
        q->push(path);
        m_frameNumber++;
    }

private:

    //  Prepare our context and socket

    std::mutex flagMutex;
};

int run ()
{
   struct sigaction sigIntHandler;

   sigIntHandler.sa_handler = my_handler;
   sigemptyset(&sigIntHandler.sa_mask);
   sigIntHandler.sa_flags = 0;

   sigaction(SIGINT, &sigIntHandler, NULL);

    // This is the data listener which will receive callbacks.  It's declared
    // before the cameraDevice so that, if this function exits with a 'return'
    // statement while the camera is still capturing, it will still be in scope
    // until the cameraDevice's destructor implicitly de-registers the listener.
    MyListener listener;
    std::queue<std::string>* Q = new std::queue<std::string>();
    listener.q = Q;

    // the camera manager will query for a connected camera
    {
        CameraManager manager;

        // if no argument was given try to open the first connected camera
        royale::Vector<royale::String> camlist (manager.getConnectedCameraList());
        cout << "Detected " << camlist.size() << " camera(s)." << endl;

        if (!camlist.empty())
        {
            cameraDevice = manager.createCamera (camlist[0]);
        }
        else
        {
            cerr << "No suitable camera device detected." << endl
                 << "Please make sure that a supported camera is plugged in, all drivers are "
                 << "installed, and you have proper USB permission" << endl;
            return 1;
        }

        camlist.clear();
    }
    // the camera device is now available and CameraManager can be deallocated here

    if (cameraDevice == nullptr)
    {
        cerr << "Cannot create the camera device" << endl;
        return 1;
    }

    // IMPORTANT: call the initialize method before working with the camera device
    auto status = cameraDevice->initialize();
    if (status != CameraStatus::SUCCESS)
    {
        cerr << "Cannot initialize the camera device, error string : " << getErrorString (status) << endl;
        return 1;
    }

    // Set use case to 10FPS
    // MODE_9_5FPS_1900
    // MODE_9_10FPS_900
    if (cameraDevice->setUseCase("MODE_9_5FPS_1900") != CameraStatus::SUCCESS)
    {
        cerr << "Error setting camera mode" << endl;
        return 1;
    }

    // register a data listener
    if (cameraDevice->registerDataListener (&listener) != CameraStatus::SUCCESS)
    {
        cerr << "Error registering data listener" << endl;
        return 1;
    }

    // start capture mode
    if (cameraDevice->startCapture() != CameraStatus::SUCCESS)
    {
        cerr << "Error starting the capturing" << endl;
        return 1;
    }

    // Initialize ZMQ
    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_REP);
    socket.bind ("tcp://*:5555");

    while (true)
    {
        usleep(1);
        if (listener.q->empty() == 1)
        {
        }
        else
        {
            // ZMQ stuff
            zmq::message_t request;
            socket.recv (&request);

            std::string msg = listener.q->front();
            cout << "From queue: " << msg << endl;

            //  Send reply back to client
            zmq::message_t reply (msg.size());
            memcpy ((void *) reply.data (), msg.c_str(), msg.size());
            socket.send (reply);

            listener.q->pop();
        }
    }

    // stop capture mode
    if (cameraDevice->stopCapture() != CameraStatus::SUCCESS)
    {
        cerr << "Error stopping the capturing" << endl;
        return 1;
    }

    return 0;
}

static PyObject*
start_camera (PyObject *dummy, PyObject *args)
{
    run();
    return NULL;
}

static struct PyMethodDef methods[] = {
    {"start_camera", start_camera, METH_VARARGS, "descript of example 2"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cModPyDem =
{
    PyModuleDef_HEAD_INIT,
    "cam", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    methods 
};

PyMODINIT_FUNC PyInit_cam(void)
{
    return PyModule_Create(&cModPyDem);
}
