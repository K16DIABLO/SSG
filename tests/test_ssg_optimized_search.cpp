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
  if (argc < 11) {
    std::cout << argv[0]
              << " data_file query_file groundtruth_file ssg_path L K result_path num_threads tau hash_bitwidth [seed]"
              << std::endl;
    exit(-1);
  }

  if (argc == 12) {
    unsigned seed = (unsigned)atoi(argv[11]);
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

  // Sungjun Jung: Setting parameters for ADA-NNS
  float tau = (float)atof(argv[9]);
  uint64_t hash_bitwidth = (uint64_t)atoi(argv[10]);
  index.SetHashBitwidth(hash_bitwidth);
  index.SetTau(tau);

  index.OptimizeGraph(data_load);

  // Sungjun Jung: Generate/Load data for ADA-NNS
  char* hash_function_name = new char[strlen(argv[1]) + strlen(".hash_function_") + strlen(argv[10]) + 1];
  char* hashed_set_name = new char[strlen(argv[1]) + strlen(".hashed_set") + strlen(argv[10]) + 1];
  strcpy(hash_function_name, argv[1]);
  strcat(hash_function_name, ".hash_function_");
  strcat(hash_function_name, argv[10]);
  strcat(hash_function_name, "b");
  strcpy(hashed_set_name, argv[1]);
  strcat(hashed_set_name, ".hashed_set_");
  strcat(hashed_set_name, argv[10]);
  strcat(hashed_set_name, "b");
  if (index.ReadHashFunction(hash_function_name)) {
    if (!index.ReadHashedSet(hashed_set_name))
      index.GenerateHashedSet(hashed_set_name, data_load);
  }
  else {
    index.GenerateHashFunction(hash_function_name);
    index.GenerateHashedSet(hashed_set_name, data_load);
  }
  delete[] hash_function_name;
  delete[] hashed_set_name;

  unsigned L = (unsigned)atoi(argv[5]);
  unsigned K = (unsigned)atoi(argv[6]);

  std::cerr << "L = " << L << ", ";
  std::cerr << "K = " << K << std::endl;

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);

  std::vector<std::vector<unsigned> > res(query_num);
  for (unsigned i = 0; i < query_num; i++) res[i].resize(K);

  auto s_q = std::chrono::high_resolution_clock::now();
  // Sungjun Jung: Hash query vector
  uint32_t* hashed_query_buffer = (uint32_t*)malloc(query_num * (hash_bitwidth >> 3));
  index.QueryHash(query_load, hashed_query_buffer, query_num);
  auto e_q = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff_query = e_q - s_q;

  // Warm up
  for (int loop = 0; loop < 3; ++loop) {
    for (unsigned i = 0; i < 10; ++i) {
      index.SearchWithOptGraph(query_load + i * dim, K, paras, res[i].data(), hashed_query_buffer + (hash_bitwidth >> 5) * i);
    }
  }

  uint32_t num_threads = (uint32_t)atoi(argv[8]);
  omp_set_num_threads(num_threads);

  auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
  for (unsigned i = 0; i < query_num; i++) {
    index.SearchWithOptGraph(query_load + i * dim, K, paras, res[i].data(), hashed_query_buffer + (hash_bitwidth >> 5) * i);
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

  std::cout << std::left << std::setw(5) << L << std::setw(10) << query_num / (diff_query.count() + diff.count()) << std::setw(10) << recall << std::endl;

  return 0;
}
