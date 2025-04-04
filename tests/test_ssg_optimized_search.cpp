//
// Created by 付聪 on 2017/6/21.
//

#include <chrono>

#include "index_random.h"
#include "index_ssg.h"
#include "util.h"
#include <omp.h>
#include <iomanip>

void save_result(char* filename, std::vector<std::vector<unsigned> >& results) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < results.size(); i++) {
    unsigned GK = (unsigned)results[i].size();
    out.write((char*)&GK, sizeof(unsigned));
    out.write((char*)results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

int main(int argc, char** argv) {
  if (argc < 9) {
    std::cout << argv[0]
              << " data_file query_file groundtruth_file ssg_path L K result_path num_threads [seed]"
              << std::endl;
    exit(-1);
  }

  if (argc == 10) {
    unsigned seed = (unsigned)atoi(argv[9]);
    srand(seed);
    std::cerr << "Using Seed " << seed << std::endl;
  }

  std::cerr << "Data Path: " << argv[1] << std::endl;

  unsigned points_num, dim;
  float* data_load = nullptr;
  data_load = efanna2e::load_data(argv[1], points_num, dim);
  data_load = efanna2e::data_align(data_load, points_num, dim);

  std::cerr << "Query Path: " << argv[2] << std::endl;

  unsigned query_num, query_dim;
  float* query_load = nullptr;
  query_load = efanna2e::load_data(argv[2], query_num, query_dim);
  query_load = efanna2e::data_align(query_load, query_num, query_dim);

  assert(dim == query_dim);

  std::cerr << "Groundtruth Path: " << argv[3] << std::endl;

  uint32_t* ground_truth_load = NULL;
  uint32_t ground_truth_num, ground_truth_dim;
  ground_truth_load = efanna2e::load_data_ivecs(argv[3], ground_truth_num, ground_truth_dim);

  efanna2e::IndexRandom init_index(dim, points_num);
  efanna2e::IndexSSG index(dim, points_num, efanna2e::FAST_L2,
                           (efanna2e::Index*)(&init_index));

  std::cerr << "SSG Path: " << argv[4] << std::endl;
  std::cerr << "Result Path: " << argv[7] << std::endl;

  index.Load(argv[4]);
  index.OptimizeGraph(data_load);

  unsigned L = (unsigned)atoi(argv[5]);
  unsigned K = (unsigned)atoi(argv[6]);

  std::cerr << "L = " << L << ", ";
  std::cerr << "K = " << K << std::endl;

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);

  std::vector<std::vector<unsigned> > res(query_num);
  for (unsigned i = 0; i < query_num; i++) res[i].resize(K);

  // Warm up
  for (int loop = 0; loop < 3; ++loop) {
    for (unsigned i = 0; i < 10; ++i) {
      index.SearchWithOptGraph(query_load + i * dim, K, paras, res[i].data());
    }
  }

  uint32_t num_threads = (uint32_t)atoi(argv[8]);
  omp_set_num_threads(num_threads);

  auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
  for (unsigned i = 0; i < query_num; i++) {
    index.SearchWithOptGraph(query_load + i * dim, K, paras, res[i].data());
  }
  auto e = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = e - s;
//  std::cerr << "Search Time: " << diff.count() << std::endl;

  save_result(argv[7], res);

  // Sungjun Jung: Evaluate Recall
  uint32_t topk_hit = 0;
  for (uint32_t i = 0; i < query_num; i++) {
    for (uint32_t j = 0; j < K; j++) {
      for (uint32_t k = 0; k < K; k++) {
        if (res[i][j] == *(ground_truth_load + i * ground_truth_dim + k)) {
          topk_hit++;
          break;
        }
      }
    }
  }
  float recall = (float)topk_hit / (query_num * K) * 100;

  std::cout << std::left << std::setw(5) << L << std::setw(10) << query_num / diff.count() << std::setw(10) << recall << std::endl;

  return 0;
}
