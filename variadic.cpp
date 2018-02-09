#include <Python.h>
#include <vector>
#include <utility>
#include <string>
#include <iostream>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

struct plotter
{
    PyObject *kwargs;
    PyObject *matplotlib;
    PyObject *pyplot;
    PyObject *callable;
    plotter()
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
        Py_DECREF(PyImport_Import(PyUnicode_FromString("matplotlib")));

        PyObject_CallMethod(matplotlib, const_cast<char*>("use"), const_cast<char*>("s"), "TkAgg");
        
        pyplot = PyImport_Import(PyUnicode_FromString("matplotlib.pyplot"));
        if (!pyplot)
            throw std::runtime_error("Error loading module pyplot!"); 

        callable = PyObject_GetAttrString(pyplot, "plot");
        if (!PyCallable_Check(callable))
            throw std::runtime_error("pyplot function is not callable!");

        Py_DECREF(PyImport_Import(PyUnicode_FromString("matplotlib.pyplot")));
    }

    ~plotter()
    {
        /* Py_Finalize(); */
    }

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
    
};

int main()
{
    std::vector<double> x{1,2,3,4,5}, y{1,2,3,4,5};
    std::pair<std::string, std::string> sp {"marker","x"};
    std::pair<std::string, double> dp {"lw",2.0};
    /* std::pair<std::string, int> ip {"s", 40}; */
    plotter p;
    p.plot(x, y, sp, dp);
}
