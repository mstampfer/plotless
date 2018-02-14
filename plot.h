#include <vector>
#include <string>
#include <utility>
#include "Plotter.h"

struct Plt : public Plotter 
{
    Plt()
    {
        callable = PyObject_GetAttrString(pyplot, "plot");
        if (!PyCallable_Check(callable))
            throw std::runtime_error("pyplot function is not callable!");
    }

    // plot y using x as index array 0..N-1
    template <typename... Args> 
    void plot(const std::vector<double>& y)
    {
        PyObject *pytup = PyTuple_New(1);
        npy_intp y_sz = static_cast<npy_intp>(y.size());
        PyTuple_SetItem(pytup, 0, PyArray_SimpleNewFromData(1, &y_sz, NPY_DOUBLE, (void*)y.data()));

        PyObject *res = PyObject_Call(callable, pytup, kwargs);
        res = PyObject_CallObject(PyObject_GetAttrString(pyplot, "show"), PyTuple_New(0));
        Py_DECREF(pytup);
        Py_DECREF(res);
    }

    template <typename... Args> 
    void plot(const std::vector<double>& y, 
            const std::pair<std::string, std::string>& p, 
            const Args& ... rest)
    {

        PyDict_SetItemString(kwargs, p.first.c_str(), PyUnicode_FromString(p.second.c_str())); 
        plot(y,rest...);
    }

    template <typename T, typename... Args> 
    void plot(const std::vector<double>& y, 
            const std::pair<std::string, T>& p, 
            const Args& ... rest)
    {

        PyDict_SetItem(kwargs, PyUnicode_FromString(p.first.c_str()), PyFloat_FromDouble(p.second)); 
        plot(y,rest...);
    }
    // end

    // plot x and y
    template <typename... Args> 
    void plot(const std::vector<double>& x, const std::vector<double>& y)
    {
        PyObject *pytup = PyTuple_New(2);
        npy_intp x_sz = static_cast<npy_intp>(x.size());
        npy_intp y_sz = static_cast<npy_intp>(y.size());
        PyTuple_SetItem(pytup, 0, PyArray_SimpleNewFromData(1, &x_sz, NPY_DOUBLE, (void*)x.data()));
        PyTuple_SetItem(pytup, 1, PyArray_SimpleNewFromData(1, &y_sz, NPY_DOUBLE, (void*)y.data())); 

        PyObject *res = PyObject_Call(callable, pytup, kwargs);
        res = PyObject_CallObject(PyObject_GetAttrString(pyplot, "show"), PyTuple_New(0));
        Py_DECREF(pytup);
        Py_DECREF(res);
    }

    template <typename... Args> 
    void plot(const std::vector<double>& x, 
            const std::vector<double>& y, 
            const std::pair<std::string, std::string>& p, 
            const Args& ... rest)
    {

        PyDict_SetItemString(kwargs, p.first.c_str(), PyUnicode_FromString(p.second.c_str())); 
        plot(x,y,rest...);
    }

    template <typename T, typename... Args> 
    void plot(const std::vector<double>& x, 
            const std::vector<double>& y, 
            const std::pair<std::string, T>& p, 
            const Args& ... rest)
    {

        PyDict_SetItem(kwargs, PyUnicode_FromString(p.first.c_str()), PyFloat_FromDouble(p.second)); 
        plot(x,y,rest...);
    }
    // end
};     
