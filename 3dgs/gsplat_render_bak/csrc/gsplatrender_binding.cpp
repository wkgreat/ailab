#define PY_SSIZE_T_CLEAN
#include "add_one.hpp"
#include "gsplatrender.hpp"
#include "numpy_utils.hpp"
#include <Python.h>
#include <cstddef>
#include <cstdlib>
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

static void free_capsule_memory(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, nullptr);
  free(ptr);
}

static int get_float_attr(PyObject *obj, const char *name, float *out) {
  PyObject *attr = PyObject_GetAttrString(obj, name);
  if (!attr) {
    return -1;
  }

  double value = PyFloat_AsDouble(attr);
  Py_DECREF(attr);
  if (PyErr_Occurred()) {
    return -1;
  }

  *out = static_cast<float>(value);
  return 0;
}

static int get_int_attr(PyObject *obj, const char *name, int *out) {
  PyObject *attr = PyObject_GetAttrString(obj, name);
  if (!attr) {
    return -1;
  }

  long value = PyLong_AsLong(attr);
  Py_DECREF(attr);
  if (PyErr_Occurred()) {
    return -1;
  }

  *out = static_cast<int>(value);
  return 0;
}

static int get_vec3_attr(PyObject *obj, const char *name, cpu::vec3f *out) {
  PyObject *vec = PyObject_GetAttrString(obj, name);
  if (!vec) {
    return -1;
  }

  int ok = get_float_attr(vec, "x", &out->x) == 0 &&
           get_float_attr(vec, "y", &out->y) == 0 &&
           get_float_attr(vec, "z", &out->z) == 0;
  Py_DECREF(vec);
  return ok ? 0 : -1;
}

static int parse_scene(PyObject *scene, SceneInfo *sceneInfo) {
  if (get_int_attr(scene, "width", &sceneInfo->viewport.width) < 0 ||
      get_int_attr(scene, "height", &sceneInfo->viewport.height) < 0 ||
      get_vec3_attr(scene, "eye", &sceneInfo->camera.eye) < 0 ||
      get_vec3_attr(scene, "center", &sceneInfo->camera.center) < 0 ||
      get_vec3_attr(scene, "up", &sceneInfo->camera.up) < 0 ||
      get_float_attr(scene, "near", &sceneInfo->projection.near) < 0 ||
      get_float_attr(scene, "far", &sceneInfo->projection.far) < 0 ||
      get_float_attr(scene, "fovy", &sceneInfo->projection.fovy) < 0) {
    return -1;
  }

  return 0;
}

static PyObject *gsplat_rasterize(PyObject *self, PyObject *args) {
  PyObject *scene = nullptr;
  PyObject *pos3d = nullptr;
  PyObject *scale = nullptr;
  PyObject *rot = nullptr;
  PyObject *opacity = nullptr;
  PyObject *sh = nullptr;

  if (!PyArg_ParseTuple(args, "OOOOOO", &scene, &pos3d, &scale, &rot, &opacity,
                        &sh)) {
    return nullptr;
  }

  PyArrayObject *arrPos3d = parseNumpyArrayF32(pos3d);
  PyArrayObject *arrScale = parseNumpyArrayF32(scale);
  PyArrayObject *arrRot = parseNumpyArrayF32(rot);
  PyArrayObject *arrOpacity = parseNumpyArrayF32(opacity);
  PyArrayObject *arrSh = parseNumpyArrayF32(sh);
  if (!arrPos3d || !arrScale || !arrRot || !arrOpacity || !arrSh) {
    Py_XDECREF(arrPos3d);
    Py_XDECREF(arrScale);
    Py_XDECREF(arrRot);
    Py_XDECREF(arrOpacity);
    Py_XDECREF(arrSh);
    return nullptr;
  }

  int nDim = 2;
  npy_intp *shape = PyArray_SHAPE(arrPos3d);
  int nsplats = shape[0];
  checkArrNDim(arrPos3d, nDim);
  checkArrNDim(arrScale, nDim);
  checkArrNDim(arrRot, nDim);
  checkArrNDim(arrOpacity, nDim);
  checkArrNDim(arrSh, nDim);
  checkArrDimShapeEqual(arrPos3d, arrScale, 0);
  checkArrDimShapeEqual(arrPos3d, arrRot, 0);
  checkArrDimShapeEqual(arrPos3d, arrOpacity, 0);
  checkArrDimShapeEqual(arrPos3d, arrSh, 0);
  checkArrDimShape(arrPos3d, 1, 3);
  checkArrDimShape(arrScale, 1, 3);
  checkArrDimShape(arrRot, 1, 4);
  checkArrDimShape(arrOpacity, 1, 1);
  checkArrDimShape(arrSh, 1, 48);

  GSplatData splats;
  splats.pos3d = static_cast<float *>(PyArray_DATA(arrPos3d));
  splats.scale = static_cast<float *>(PyArray_DATA(arrScale));
  splats.rot = static_cast<float *>(PyArray_DATA(arrRot));
  splats.opacity = static_cast<float *>(PyArray_DATA(arrOpacity));
  splats.sh = static_cast<float *>(PyArray_DATA(arrSh));
  splats.nsplats = nsplats;

  SceneInfo sceneInfo;
  if (parse_scene(scene, &sceneInfo) < 0) {
    Py_DECREF(arrPos3d);
    Py_DECREF(arrScale);
    Py_DECREF(arrRot);
    Py_DECREF(arrOpacity);
    Py_DECREF(arrSh);
    return nullptr;
  }

  float *image = nullptr;
  // gsplat_rasterize_gpu(splats, sceneInfo, &image);
  image = (float *)malloc(sceneInfo.viewport.width * sceneInfo.viewport.height *
                          4 * sizeof(float));

  Py_DECREF(arrPos3d);
  Py_DECREF(arrScale);
  Py_DECREF(arrRot);
  Py_DECREF(arrOpacity);
  Py_DECREF(arrSh);

  npy_intp dim[2] = {sceneInfo.viewport.height, sceneInfo.viewport.width};

  PyObject *imageArr = PyArray_SimpleNewFromData(2, dim, NPY_FLOAT32, image);
  if (!imageArr) {
    free(image);
    return nullptr;
  }

  PyObject *capsule = PyCapsule_New(image, nullptr, free_capsule_memory);
  if (!capsule) {
    Py_DECREF(imageArr);
    free(image);
    return nullptr;
  }

  if (PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(imageArr),
                            capsule) < 0) {
    Py_DECREF(imageArr);
    Py_DECREF(capsule);
    return nullptr;
  }

  return imageArr;
}

static PyMethodDef Methods[] = {
    {"add_one", numpy_add_one, METH_VARARGS, "Example NumPy C++ function"},
    {"gsplat_rasterize", gsplat_rasterize, METH_VARARGS,
     "Gaussian Splats Rasterizer"},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_C", nullptr, -1,
                                    Methods};

PyMODINIT_FUNC PyInit__C(void) {
  import_array();
  return PyModule_Create(&module);
}
