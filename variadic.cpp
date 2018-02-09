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
        wchar_t name[] = L"variadic";
        Py_SetProgramName(name);
        Py_Initialize();
        if (!Py_IsInitialized())
        {
            throw std::runtime_error("Error initializing Python interpreter\n");
        }
        PyObject *sys = PyImport_ImportModule("sys");
        auto xxx = PySys_GetObject("path");
        auto zzz = PyImport_ImportModule("numpy.core.multiarray");
        int x = _import_array();
        if (x<0)
            throw std::runtime_error("Error loading module multiarray!"); 

        /* PyObject *yy = PyUnicode_FromString("matplotlib"); */
        matplotlib = PyImport_ImportModule("matplotlib");
        if (!matplotlib)
            throw std::runtime_error("Error loading module matplotlib!"); 
        Py_DECREF(PyImport_Import(PyUnicode_FromString("matplotlib")));
        PyObject_CallMethod(matplotlib, const_cast<char*>("use"), const_cast<char*>("s"), "TkAgg");
        
        PyObject *xx = PyUnicode_FromString("matplotlib.pyplot");
        pyplot = PyImport_Import(xx);
        if (!pyplot)
            throw std::runtime_error("Error loading module pyplot!"); 
        callable = PyObject_GetAttrString(pyplot, "plot");
        if (!PyCallable_Check(callable))
            throw std::runtime_error("pyplot function is not callable!");
        Py_DECREF(PyImport_Import(PyUnicode_FromString("matplotlib.pyplot")));
        kwargs = PyDict_New();
    }

    ~plotter()
    {
        /* Py_Finalize(); */
    }

    template <typename... Args> 
    void plot(const std::vector<double>& x, const std::vector<double>& y)
    {
        PyObject *pytup = PyTuple_New(2);
        npy_intp sz_x = x.size();
        npy_intp sz_y = y.size();
        std::vector<double> vx_dup;
        vx_dup.reserve(x.size());
        std::copy(x.begin(), x.end(), std::back_inserter(vx_dup));
        std::vector<double> vy_dup;
        vy_dup.reserve(y.size());
        std::copy(y.begin(), y.end(), std::back_inserter(vy_dup));
        auto xa = PyArray_SimpleNewFromData(1, &sz_x, NPY_DOUBLE, (void*)vx_dup.data());
        auto ya = PyArray_SimpleNewFromData(1, &sz_y, NPY_DOUBLE, (void*)vy_dup.data());
        PyTuple_SetItem(pytup, 0, xa);
        PyTuple_SetItem(pytup, 1, ya); 
        if (!PyCallable_Check(callable))
            throw std::runtime_error("pyplot function is not callable!");
        PyObject *res = PyObject_Call(callable, pytup, kwargs);
        auto tup = PyTuple_New(0);
        auto show = PyObject_GetAttrString(pyplot, "show");

        res = PyObject_CallObject(show,tup);
    }

    template <typename... Args> 
    void plot(const std::vector<double>& x, 
            const std::vector<double>& y, 
            const std::pair<std::string, std::string>& p, 
            const Args& ... rest)
    {

        std::cout << "in plot" << std::endl;
        PyDict_SetItemString(kwargs, p.first.c_str(), PyUnicode_FromString(p.second.c_str())); 
        plot(x,y,rest...);
    }

    template <typename T, typename... Args> 
    void plot(const std::vector<double>& x, 
            const std::vector<double>& y, 
            const std::pair<std::string, T>& p, 
            const Args& ... rest)
    {

        std::cout << "in plot" << std::endl;
        PyDict_SetItem(kwargs, PyUnicode_FromString(p.first.c_str()), PyFloat_FromDouble(p.second)); 
        plot(x,y,rest...);
    }
    
};

int main()
{
    std::vector<double> x{1,2,3,4,5}, y{1,2,3,4,5};
    std::pair<std::string, std::string> sp {"marker","x"};
    /* std::pair<std::string, double> dp {"lw",0.5}; */
    /* std::pair<std::string, int> ip {"s", 40}; */
    plotter p;
    p.plot(x, y, sp);
}
