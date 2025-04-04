#ifndef EFANNA2E_UTIL_H
#define EFANNA2E_UTIL_H

#include <random>

namespace efanna2e {

void GenRandom(std::mt19937& rng, unsigned* addr, unsigned size, unsigned N);

float* load_data(const char* filename, unsigned& num, unsigned& dim);

uint32_t* load_data_ivecs(const char* filename, uint32_t& num, uint32_t& dim);

float* data_align(float* data_ori, unsigned point_num, unsigned& dim);

}  // namespace efanna2e

#endif  // EFANNA2E_UTIL_H
