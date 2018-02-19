#include <iostream>
#include <Python.h>
#include <vector>
#include <utility>
#include <string>


template <typename T>
struct Args 
{
    std::vector<T> v;
    PyObject *args;

    ~Args()
    {
        Py_DECREF(args);
    }
    template<typename U>
    Args(std::initializer_list<U>& il) : v(il)
    {
        int i = 0;
        args = PyTuple_New(v.size());
        for (const auto& e: v)
        {
            std::cout << e <<std::endl;
            auto y_sz = static_cast<npy_intp>(e.size());
            PyTuple_SetItem(args, i, PyArray_SimpleNewFromData(1, &y_sz, NPY_DOUBLE, (void*)e.data()));
            ++i;
        }
    }
    template<typename U>
    Args(std::vector<U>& vec) : v(vec)
    {
        int i = 0;
        args = PyTuple_New(v.size());
        for (const auto& e: v)
        {
            std::cout << e <<std::endl;
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
           const Args& ... rest)
    {
        PyDict_SetItemString(kwargs, p.first, PyUnicode_FromString(p.second)); 
        Kwargs(rest...);
    }

    template <typename T> 
    Kwargs(const std::pair<const char*, T>& p) 
    {

        PyDict_SetItem(kwargs, PyUnicode_FromString(p.first), PyFloat_FromDouble(p.second)); 
    }

    template <typename T, typename... Args> 
    Kwargs(const std::pair<const char*, T>& p, 
           const Args& ... rest)
    {

        PyDict_SetItem(kwargs, PyUnicode_FromString(p.first), PyFloat_FromDouble(p.second)); 
        Kwargs(rest...);
    }
};


