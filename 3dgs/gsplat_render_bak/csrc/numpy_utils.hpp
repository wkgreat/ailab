
#pragma once
#include <numpy/arrayobject.h>
#include <stdexcept>

static inline PyArrayObject *parseNumpyArrayF32(PyObject *obj) {
  PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));

  if (!arr) {
    return nullptr;
  }

  return arr;
}

static inline void checkArrNDim(PyArrayObject *arr, int nDim) {
  if (PyArray_NDIM(arr) != nDim) {
    throw std::runtime_error("wrong nDim of numpy array!");
  }
};

static inline void checkArrDimShape(PyArrayObject *arr, int dim, int size) {
  npy_intp *s = PyArray_SHAPE(arr);
  if (s[dim] != size) {
    throw std::runtime_error("wrong numpy array shape");
  }
}

static inline void checkArrDimShapeEqual(PyArrayObject *arr0,
                                         PyArrayObject *arr1, int dim) {

  npy_intp *s0 = PyArray_SHAPE(arr0);
  npy_intp *s1 = PyArray_SHAPE(arr1);
  if (s0[dim] != s1[dim]) {
    throw std::runtime_error("shape in not equal!");
  }
}
