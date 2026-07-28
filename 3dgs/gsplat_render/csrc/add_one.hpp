#pragma once

#include <cstddef>

void add_one_cpu(const float *data, float *result, const size_t size);
void add_one_gpu(const float *data, float *result, const int n);