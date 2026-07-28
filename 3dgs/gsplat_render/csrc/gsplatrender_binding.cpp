#define PY_SSIZE_T_CLEAN
#include "gsplatrender.hpp"
#include <Python.h>
#include <cstddef>
#include <numpy/arrayobject.h>

static PyObject *numpy_add_one(PyObject *self, PyObject *args) {
  PyObject *input_obj;

  if (!PyArg_ParseTuple(args, "O", &input_obj)) {
    return nullptr;
  }

  PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(input_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));

  if (!arr) {
    return nullptr;
  }

  // 使用 NumPy 数据
  float *data = static_cast<float *>(PyArray_DATA(arr));
  int ndim = PyArray_NDIM(arr);
  npy_intp *np_shape = PyArray_SHAPE(arr);
  npy_intp size = PyArray_SIZE(arr);

  PyObject *result_arr = PyArray_EMPTY(ndim, np_shape, NPY_FLOAT32, 0);
  if (!result_arr) {
    Py_DECREF(arr);
    return nullptr;
  }

  float *result = static_cast<float *>(
      PyArray_DATA(reinterpret_cast<PyArrayObject *>(result_arr)));
  add_one_gpu(data, result, static_cast<size_t>(size));

  Py_DECREF(arr);

  return result_arr;
}

static PyMethodDef Methods[] = {
    {"add_one", numpy_add_one, METH_VARARGS, "Example NumPy C++ function"},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_C", nullptr, -1,
                                    Methods};

PyMODINIT_FUNC PyInit__C(void) {
  import_array();
  return PyModule_Create(&module);
}
