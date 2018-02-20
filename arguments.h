#include <iostream>
#include <Python.h>
#include <vector>
#include <utility>
#include <string>


template <typename T>
struct Args 
{
    PyObject *args;

    ~Args()
    {
        Py_DECREF(args);
    }
    template<typename U>
    Args(std::vector<U>& v)
    {
        int i = 0;
        args = PyTuple_New(v.size());
        for (const auto& e: v)
        {
            auto y_sz = static_cast<npy_intp>(e.size());
            PyTuple_SetItem(args, i, PyArray_SimpleNewFromData(1, &y_sz, NPY_DOUBLE, (void*)e.data()));
            ++i;
        }
    }

    Args(std::vector<double>&& v) 
    {
        int i = 0;
        args = PyTuple_New(v.size());
        for (const auto& e: v)
        {
            PyTuple_SetItem(args, i, PyFloat_FromDouble(e)); 
            ++i;
        }
    }
};

template <typename... Types>
struct Kwargs 
{
    PyObject *kwargs = PyDict_New();

    ~Kwargs()
    {
        Py_DECREF(kwargs);
    }

    Kwargs(const std::pair<const char*, const char*>& p) 
    {
        PyDict_SetItemString(kwargs, p.first, PyUnicode_FromString(p.second)); 
    }

    template <typename... Args> 
    Kwargs(const std::pair<const char*, const char*>& p, 
           const Args& ... rest) : Kwargs(rest...)
    {
        PyDict_SetItemString(kwargs, p.first, PyUnicode_FromString(p.second)); 
    }

    template <typename T> 
    Kwargs(const std::pair<const char*, T>& p) 
    {
        PyDict_SetItem(kwargs, PyUnicode_FromString(p.first), PyFloat_FromDouble(p.second)); 
    }

    template <typename T, typename... Args> 
    Kwargs(const std::pair<const char*, T>& p, 
           const Args& ... rest) : Kwargs(rest...)
    {
        PyDict_SetItem(kwargs, PyUnicode_FromString(p.first), PyFloat_FromDouble(p.second)); 
    }
};


