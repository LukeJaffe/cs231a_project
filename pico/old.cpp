#include <iostream>
#include "Python.h"
#include "numpy/arrayobject.h"


#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <fstream>
#include <sstream>
#include <string>

#include <royale.hpp>
#include <royale/IPlaybackStopListener.hpp>
#include <royale/IReplay.hpp>

using namespace royale;
using namespace std;

namespace
{
    class MyListener : public IDepthDataListener
    {
    public:
        MyListener()
        {
            // reset the current frame number
            m_frameNumber = 0;
        }

        void parse_frame (const royale::DepthData *data)
        {
            float    *buffer = (float*)array->data;

            int k = 0;
            int frame_offset = m_frameNumber*data->width*data->height;
            int mode_offset = numFrames*data->width*data->height;
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
                        //cout << g << " | " << z << endl;
                        buffer[frame_offset+k] = g;
                        buffer[mode_offset+frame_offset+k] = z;
                    }
                }
            }
        }

        void onNewData (const DepthData *data)
        {

            cout << "Exporting frame " << m_frameNumber << " of " << numFrames << endl;

            parse_frame (data);
            // Increment frame number, important for indexing
            m_frameNumber++;
        }

        uint32_t numFrames;     // Total number of frames in the recording
        string rrfFile;         // Recording file that was opened
        PyArrayObject *array;

    private:
        uint32_t m_frameNumber; // the current frame number
    };

    class MyPlaybackStopListener : public IPlaybackStopListener
    {
    public:
        MyPlaybackStopListener()
        {
            playbackRunning = true;
        }

        void onPlaybackStopped()
        {
            lock_guard<mutex> lock (m_stopMutex);
            playbackRunning = false;
        }

        void waitForStop()
        {
            bool running = true;
            do
            {
                {
                    lock_guard<mutex> lock (m_stopMutex);
                    running = playbackRunning;
                }

                this_thread::sleep_for (chrono::milliseconds (50));
            }
            while (running);
        }

    private:
        mutex m_stopMutex;      // Mutex to synchronize the access to playbackRunning
        bool playbackRunning;   // Shows if the playback is still running
    };

}

PyArrayObject* parse_video (char* fname)
{
    // This is the data listener which will receive callbacks.  It's declared
    // before the cameraDevice so that, if this function exits with a 'return'
    // statement while the camera is still capturing, it will still be in scope
    // until the cameraDevice's destructor implicitly deregisters the listener.
    MyListener listener;

    // PlaybackStopListener which will be called as soon as the playback stops.
    MyPlaybackStopListener stopListener;

    // this represents the main camera device object
    unique_ptr<ICameraDevice> cameraDevice;

    // check the command line for a given file
    // the camera manager can also be used to open recorded files
    CameraManager manager;

    // create a device from the file
    cameraDevice = manager.createCamera (fname);

    // if the file was loaded correctly the cameraDevice is now available
    if (cameraDevice == nullptr)
    {
        cerr << "Cannot load the file " << fname << endl;
        return NULL;
    }

    listener.rrfFile = fname;

    // cast the cameraDevice to IReplay which offers more options for playing
    // back recordings
    IReplay *replayControls = dynamic_cast<IReplay *> (cameraDevice.get());

    if (replayControls == nullptr)
    {
        cerr << "Unable to cast to IReplay interface" << endl;
        return NULL;
    }

    // IMPORTANT: call the initialize method before working with the camera device
    if (cameraDevice->initialize() != CameraStatus::SUCCESS)
    {
        cerr << "Cannot initialize the camera device" << endl;
        return NULL;
    }

    // turn off the looping of the playback
    replayControls->loop (false);

    // turn off the timestamps (this will speed up the conversion)
    replayControls->useTimestamps (false);

    // retrieve the total number of frames from the recording
    listener.numFrames = replayControls->frameCount();
    npy_intp dim[4]= {2, listener.numFrames, 287, 352};
    // create the array
    listener.array = (PyArrayObject *) PyArray_SimpleNew(4, dim, PyArray_FLOAT);

    // register a data listener
    if (cameraDevice->registerDataListener (&listener) != CameraStatus::SUCCESS)
    {
        cerr << "Error registering data listener" << endl;
        return NULL;
    }

    // register a playback stop listener. This will be called as soon
    // as the file has been played back once (because loop is turned off)
    replayControls->registerStopListener (&stopListener);

    // start capture mode
    if (cameraDevice->startCapture() != CameraStatus::SUCCESS)
    {
        cerr << "Error starting the capturing" << endl;
        return NULL;
    }

    // block until the playback has finished
    stopListener.waitForStop();

    // stop capture mode
    if (cameraDevice->stopCapture() != CameraStatus::SUCCESS)
    {
        cerr << "Error stopping the capturing" << endl;
        return NULL;
    }

    return listener.array;
}

static PyObject*
f1 (PyObject *dummy, PyObject *args)
{
    PyObject *arg1=NULL;
    PyObject *arr1=NULL;
    int nd;

    if (!PyArg_ParseTuple(args, "O", &arg1))
        return NULL;

    arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr1 == NULL)
        return NULL;

    nd = PyArray_NDIM(arr1);   //number of dimensions

    Py_DECREF(arr1);

    return PyLong_FromLong(nd);
}

static PyObject*
f2 (PyObject *dummy, PyObject *args)
{
    char* s;

    if (!PyArg_ParseTuple(args, "s", &s)) return NULL;

    std::cout<< s <<std::endl;

    PyArrayObject *array = parse_video(s);

    return PyArray_Return((PyArrayObject *)array);
}

static struct PyMethodDef methods[] = {
    {"f1", f1, METH_VARARGS, "descript of example 1"},
    {"f2", f2, METH_VARARGS, "descript of example 2"},
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
    import_array();
    return PyModule_Create(&cModPyDem);
}
