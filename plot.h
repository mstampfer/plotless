#include <iostream>
#include <Python.h>
#include <vector>
#include "plotter.h"


struct Plt : public Plotter 
{
    PyObject *callable;

    Plt()
    {
        callable = PyObject_GetAttrString(pyplot, "plot");
        if (!PyCallable_Check(callable))
            throw std::runtime_error("pyplot function is not callable!");
    }

    /* template < typename... Types1, template <typename...> class T */
             /* , typename... Types2, template <typename...> class V> */
    /* void plot(const T<Types1...>& Args, const V<Types2...>& Kwargs) */
    void plot(PyObject* args, PyObject* kwargs)
    {
        PyObject *res = PyObject_Call(callable, args, kwargs);
        res = PyObject_CallObject(PyObject_GetAttrString(pyplot, "show"), PyTuple_New(0));
    }

    ~Plt()
    {
        Py_DECREF(callable);
    }
};

struct Parameters
{
    PyObject *kwargs;
    PyObject *args;
    Parameters()
    {
        kwargs = PyDict_New();
    }
    virtual ~Parameters()
    {
        Py_DECREF(args);
        Py_DECREF(kwargs);
    }

};

template <typename T>
struct Args : public Parameters
{
    std::vector<T> v;

    template<typename U>
    Args(std::initializer_list<U>& il) : v(il)
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
};

template <typename... Types>
struct Kwargs : public Parameters
{
    Kwargs(const std::pair<std::string, std::string>& p) 
    {
        PyDict_SetItemString(kwargs, p.first.c_str(), PyUnicode_FromString(p.second.c_str())); 
    }

    template <typename... Args> 
    Kwargs(const std::pair<std::string, std::string>& p, 
           const Args& ... rest)
    {
        PyDict_SetItemString(kwargs, p.first.c_str(), PyUnicode_FromString(p.second.c_str())); 
        Kwargs(rest...);
    }

    template <typename T> 
    Kwargs(const std::pair<std::string, T>& p) 
    {

        PyDict_SetItem(kwargs, PyUnicode_FromString(p.first.c_str()), PyFloat_FromDouble(p.second)); 
    }

    template <typename T, typename... Args> 
    Kwargs(const std::pair<std::string, T>& p, 
           const Args& ... rest)
    {

        PyDict_SetItem(kwargs, PyUnicode_FromString(p.first.c_str()), PyFloat_FromDouble(p.second)); 
        Kwargs(rest...);
    }
};


        
