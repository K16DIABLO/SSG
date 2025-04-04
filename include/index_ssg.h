#ifndef EFANNA2E_INDEX_SSG_H
#define EFANNA2E_INDEX_SSG_H

#include <boost/dynamic_bitset.hpp>
#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>

#include "index.h"
#include "neighbor.h"
#include "parameters.h"
#include "util.h"

#include <Eigen/Dense>

namespace efanna2e {

class IndexSSG : public Index {
 public:
  explicit IndexSSG(const size_t dimension, const size_t n, Metric m,
                    Index *initializer);

  virtual ~IndexSSG();

  virtual void Save(const char *filename) override;
  virtual void Load(const char *filename) override;

  virtual void Build(size_t n, const float *data,
                     const Parameters &parameters) override;

  virtual void Search(const float *query, const float *x, size_t k,
                      const Parameters &parameters, unsigned *indices) override;
  void SearchWithOptGraph(const float *query, size_t K,
                          const Parameters &parameters, unsigned *indices,
                          uint32_t* hashed_query_buffer);
  void OptimizeGraph(const float *data);

  // Sungjun Jung: ADA-NNS functions
  void SetTau(const float tau) { tau_ = tau; }
  void SetHashBitwidth(const uint64_t hash_bitwidth) { hash_bitwidth_ = hash_bitwidth; }
  void GenerateHashFunction (char* file_name);
  void GenerateHashedSet (char* file_name, float* data);
  bool ReadHashFunction (char* file_name);
  bool ReadHashedSet (char* file_name);
  void QueryHash(const float* query, uint32_t* hashed_query, const uint64_t num_query);
  uint32_t CandidateSelection(const float query_norm, const uint32_t* hashed_query, std::vector<SimpleNeighbor>& selected_pool, boost::dynamic_bitset<>& flags, const uint32_t* neighbors);

 protected:
  typedef std::vector<std::vector<unsigned>> CompactGraph;
  typedef std::vector<SimpleNeighbors> LockGraph;
  typedef std::vector<nhood> KNNGraph;

  CompactGraph final_graph_;
  Index *initializer_;

  void init_graph(const Parameters &parameters);
  void get_neighbors(const float *query, const Parameters &parameter,
                     std::vector<Neighbor> &retset,
                     std::vector<Neighbor> &fullset);
  void get_neighbors(const unsigned q, const Parameters &parameter,
                     std::vector<Neighbor> &pool);
  void sync_prune(unsigned q, std::vector<Neighbor> &pool,
                  const Parameters &parameter, float threshold,
                  SimpleNeighbor *cut_graph_);
  void Link(const Parameters &parameters, SimpleNeighbor *cut_graph_);
  void InterInsert(unsigned n, unsigned range, float threshold,
                   std::vector<std::mutex> &locks, SimpleNeighbor *cut_graph_);
  void Load_nn_graph(const char *filename);
  void strong_connect(const Parameters &parameter);

  void DFS(boost::dynamic_bitset<> &flag,
           std::vector<std::pair<unsigned, unsigned>> &edges, unsigned root,
           unsigned &cnt);
  bool check_edge(unsigned u, unsigned t);
  void findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                const Parameters &parameter);
  void DFS_expand(const Parameters &parameter);

 private:
  unsigned width;
  unsigned ep_; //not in use
  std::vector<unsigned> eps_;
  std::vector<std::mutex> locks;
  char *opt_graph_;
  size_t node_size;
  size_t data_len;
  size_t neighbor_len;
  KNNGraph nnd_graph;

  // Sungjun Jung: Variables for ADA-NNS
  float tau_; // candidate selection threshold
              // _tau = 0.3 means only top 30% of neighbors 
              // having the smallest angular distance to the query are selected
  uint32_t hash_bitwidth_;
  float* hash_function_;
  uint32_t* hashed_set_;
};

}  // namespace efanna2e

#endif  // EFANNA2E_INDEX_SSG_H
