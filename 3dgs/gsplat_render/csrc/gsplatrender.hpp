#pragma once

#include "matrix.hpp"

using namespace gsplat::matrix;

typedef struct GSplatData {
  float *pos3d;
  float *scale;
  float *rot;
  float *opacity;
  float *sh;
  int nsplats;
};

typedef struct Viewport {
  int width;
  int height;
};

typedef struct CameraInfo {
  cpu::vec3f eye;
  cpu::vec3f center;
  cpu::vec3f up;
};

typedef struct ProjectionInfo {
  float near;
  float far;
  float fovy;
};

typedef struct SceneInfo {
  Viewport viewport;
  CameraInfo camera;
  ProjectionInfo projection;
};
