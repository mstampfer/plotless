#include <Python.h>
#include <vector>
#include <utility>
#include <string>
#include <iostream>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

struct Plotter
{
    PyObject *matplotlib;
    PyObject *pyplot;
    PyObject *kwargs;
    PyObject *callable;

    Plotter()
    {
        kwargs = PyDict_New();
        wchar_t name[] = L"variadic";
        Py_SetProgramName(name);
        Py_Initialize();
        if (!Py_IsInitialized())
        {
            throw std::runtime_error("Error initializing Python interpreter\n");
        }
        PyObject *sys = PyImport_ImportModule("sys");
        if (_import_array()<0)
            throw std::runtime_error("Error loading module multiarray!"); 

        matplotlib = PyImport_ImportModule("matplotlib");
        if (!matplotlib)
            throw std::runtime_error("Error loading module matplotlib!"); 

        PyObject_CallMethod(matplotlib, const_cast<char*>("use"), const_cast<char*>("s"), "TkAgg");
        
        pyplot = PyImport_Import(PyUnicode_FromString("matplotlib.pyplot"));
        if (!pyplot)
            throw std::runtime_error("Error loading module pyplot!"); 

    }

    template<typename...Args>
    virtual void plot(const std::vector<double> &x, const std::vector<double> &y,
            const Args&... rest) = 0;

    ~Plotter()
    {
        Py_DECREF(callable);
        Py_DECREF(kwargs);
        Py_DECREF(pyplot);
        Py_DECREF(matplotlib);
        Py_Finalize();
    }
    
    
};

