#pragma once

#include "matrix.hpp"

using namespace gsplat::matrix;

struct GSplatData {
  float *pos3d;
  float *scale;
  float *rot;
  float *opacity;
  float *sh;
  int nsplats;
};

struct Viewport {
  int width;
  int height;
};

struct CameraInfo {
  cpu::vec3f eye;
  cpu::vec3f center;
  cpu::vec3f up;
};

struct ProjectionInfo {
  float near;
  float far;
  float fovy;
};

struct SceneInfo {
  Viewport viewport;
  CameraInfo camera;
  ProjectionInfo projection;
};

void gsplat_rasterize_gpu(GSplatData splats, SceneInfo scene,
                          float **out_image);
