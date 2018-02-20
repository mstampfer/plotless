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
    PyObject *xlabel_f;
    PyObject *ylabel_f;
    PyObject *grid_f;

    Plt()
    {
        if (!PyCallable_Check(plot_f = PyObject_GetAttrString(pyplot, "plot")))
            throw std::runtime_error("plot function is not callable!");

        if (!PyCallable_Check(axis_f=PyObject_GetAttrString(pyplot, "axis")))
            throw std::runtime_error("axis function is not callable!");

        if (!PyCallable_Check(xlabel_f=PyObject_GetAttrString(pyplot, "xlabel")))
            throw std::runtime_error("xlabel function is not callable!");

        if (!PyCallable_Check(ylabel_f=PyObject_GetAttrString(pyplot, "ylabel")))
            throw std::runtime_error("ylabel function is not callable!");

        if (!PyCallable_Check(grid_f=PyObject_GetAttrString(pyplot, "grid")))
            throw std::runtime_error("grid function is not callable!");
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

    void xlabel(const char *s)
    {
        PyObject* args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, PyUnicode_FromString(s));
        PyObject *xlabel = PyObject_Call(xlabel_f, args, NULL);
        if (!xlabel)
        {
            std::cout << "\nxlabel: ";
            PyObject_Print(PyUnicode_FromString(s),stdout,0);
            throw std::runtime_error("Unable to call xlabel with above arguments.");
        }

    }

    void ylabel(const char *s)
    {
        PyObject* args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, PyUnicode_FromString(s));
        PyObject *ylabel = PyObject_Call(ylabel_f, args, NULL);
        if (!ylabel)
        {
            std::cout << "\nylabel: ";
            PyObject_Print(PyUnicode_FromString(s),stdout,0);
            throw std::runtime_error("Unable to call ylabel with above arguments.");
        }

    }

    void grid(bool b)
    {
        PyObject* args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, PyBool_FromLong(b));
        PyObject *grid = PyObject_Call(grid_f, args, NULL);
        if (!grid)
        {
            std::cout << "\ngrid: ";
            PyObject_Print(PyBool_FromLong(b),stdout,0);
            throw std::runtime_error("Unable to call grid with above arguments.");
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
        Py_DECREF(xlabel_f);
        Py_DECREF(ylabel_f);
        Py_DECREF(grid_f);
    }
};
