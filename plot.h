#include <iostream>
#include <Python.h>
#include <vector>
#include "plotter.h"
#include "arguments.h"
#include <Eigen/Dense>
using namespace Eigen;

struct Plt : public Plotter 
{
    PyObject *plot_f;
    PyObject *axis_f;

    Plt()
    {
        plot_f = PyObject_GetAttrString(pyplot, "plot");
        if (!PyCallable_Check(plot_f))
            throw std::runtime_error("plot function is not callable!");
        axis_f = PyObject_GetAttrString(pyplot, "axis");
        if (!PyCallable_Check(axis_f))
            throw std::runtime_error("axis function is not callable!");
    }

    template <typename T, typename U>
    void plot(const T& a, const U& b)
    { 
        PyObject *plot = PyObject_Call(plot_f, a.args, b.kwargs);
        if (!plot)
        {
            std::cout << "\nargs: ";
            PyObject_Print(a.args,stdout,0);
            std::cout<< "\nkwargs: ";
            PyObject_Print(b.kwargs,stdout,0);
            std::cout<<std::endl;
            throw std::runtime_error("Unable to call plot with above arguments.");
        }
    }

    void axis(const std::vector<double>& v) 
    {
        PyObject* args = PyTuple_New(1);
        auto y_sz = static_cast<npy_intp>(v.size());
        PyTuple_SetItem(args, 0, PyArray_SimpleNewFromData(1, &y_sz, NPY_DOUBLE, (void*)v.data()));
        PyObject *axis = PyObject_Call(axis_f, args, NULL);
        if (!axis)
        {
            std::cout << "\nargs: ";
            PyObject_Print(args,stdout,0);
            throw std::runtime_error("Unable to call axis with above arguments.");
        }
            
    }

    void show()
    {
       PyObject_CallObject(PyObject_GetAttrString(pyplot, "show"), PyTuple_New(0));
    }

    ~Plt()
    {
        Py_DECREF(plot_f);
        Py_DECREF(axis_f);
    }
};
