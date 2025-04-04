# SSG : Satellite System Graph For Approximate Nearest Neighbor Search

### Prerequisites

+ GCC 9.4.0+ with OpenMP
+ CMake 3.22.2+
+ Boost 1.55+
+ [TCMalloc](http://goog-perftools.sourceforge.net/doc/tcmalloc.html)

### Datasets

| Name     | Dimension | No. of base | No. of query | Metric |
|----------|-----------|-------------|--------------|--------|
| [SIFT1M](http://corpus-texmex.irisa.fr/)   | 128       | 1,000,000   | 10,000       | L2 |
| [GIST1M](http://corpus-texmex.irisa.fr/)   | 960       | 1,000,000   | 1,000        | L2 |
| [CRAWL](http://github.com/ZJULearning/SSG)    | 300       | 1,989,995   | 10,000       | L2 |
| DEEP100M* | 96        | 100,000,000 | 10,000        | L2 |
+ For DEEP100M, we will share the file link upon request

### Dataset Conversion

For datasets provided in HDF5 format (e.g., GLOVE-100 and NYTIMES),  
Parse HDF5 and generate fvecs and ivecs as follows:  
```bash
python ./utils/hdf5_to_vecs.py [hdf5_file_name]
mkdir -p dataset/[dataset_name]
mv [dataset_name]_*vecs dataset/[dataset_name]
```

### Compile On Ubuntu

Install Dependencies:

```bash
sudo apt-get install g++ cmake libgoogle-perftools-dev
```

Compile SSG:
```bash
git submodule update --init --recursive
mkdir build
cd build
cmake .. && make -j
```

### Million-scale Tests Reproduction
Download datasets in the `dataset` directory

Build kNN Graph:  
You can use either [efanna\_graph](https://github.com/ZJULearning/efanna\_graph) or [faiss](https://github.com/facebookresearch/faiss) to build this kNN graph.

The parameters used [faiss](https://github.com/facebookresearch/faiss) to build each kNN graph is as follows:

| Dataset | K |
|---------|---|
| SIFT1M  | 200 |
| GIST1M  | 400 |
| CRAWL   | 400 |
| DEEP100M | 400 |

Build SSG index from kNN Graph:  
You can use following command to build SSG index:

```bash
cd [SSG_HOME]/build/tests
./test_ssg_index [dataset_path] [kNN_graph_path] [L] [R] [Angle] [ssg_index_path]
```
* **L** controls the quality of the SSG, the larger the better, L > R.
* **R** controls the index size of the graph, the best R is related to the intrinsic dimension of the dataset.
* **Angle** controls the angle between two edges.

| Dataset          | L   | R     | Angle|
|----------|-----------|-------------|--------------|
| SIFT1M      | 100 | 50   | 60    |
| GIST1M      | 500 | 70   | 60    |
| CRAWL       | 500 | 40   | 60    |
| DEEP100M    | 500 | 40   | 60    |

These are parameters used to build SSG index.

To reproduce ADA-NNS (SSG) results:
```bash
cd [SSG_HOME]/build/tests
./test_ssg_optimized_search [dataset_path] [query_path] [groundtruth_path] [ssg_index_path] [search_L] [search_K] [result_path] [num_threads] [random_seed (optional)]
```
* **search\_L** controls the quality of the search results, the larger the better but slower (must larger than search\_K).
* **search\_K** controls the number of neighbors we want to find.
* **random\_seed (optional)** is the random seed.

> **NOTE:** Data alignment is essential for the correctness of our procedure, because we use SIMD instructions for acceleration of numerical computing such as AVX and SSE2.
You should use it to ensure your data elements (feature) is aligned with 8 or 16 int or float. For example, if your features are of dimension 70, then it should be extend to dimension 72. And the last 2 dimension should be filled with 0 to ensure the correctness of the distance computing. And this is what data\_align() does.

> **NOTE:** Only data-type int32 and float32 are supported for now.
