# A Comparative Analysis of Approximate Nearest Neighbor Methods for Entity Resolution

This repository contains the source code, datasets, and experimental results for the paper: "A Comparative Analysis of Approximate Nearest Neighbor Methods On Dense Embeddings for Entity Resolution".

## 📖 Abstract

Entity Resolution (ER) is a foundational task in data integration, and modern approaches increasingly rely on converting entity data into dense vector embeddings. The critical blocking step is thus transformed into an Approximate Nearest Neighbor (ANN) search problem. While many deep learning-based ER systems leverage ANN methods, the specific choice of indexing algorithm is often underspecified, despite the profound impact this choice has on performance. This paper addresses this gap by providing a rigorous and comprehensive experimental evaluation of nine influential ANN algorithms from all major families (graph-based, inverted-index, and hashing). A primary contribution is a detailed, quantitative analysis of the impact of vector quantization—specifically Product Quantization (PQ) and Scalar Quantization (SQ)—on the performance profiles of baseline algorithms. Our results establish a clear performance hierarchy, demonstrating the superiority of graph-based and inverted-file methods, the critical role of quantization for scalability, and the unsuitability of LSH for large-scale tasks. The analysis culminates in a practical decision framework to assist researchers and practitioners in selecting the most appropriate ANN indexing strategy based on their specific application constraints.

## 🚀 Key Contributions

* **Rigorous Benchmarking:** An extensive experimental comparison of nine state-of-the-art ANN algorithms across eight diverse datasets.
* **In-depth Quantization Analysis:** A detailed quantitative analysis of the impact of Product Quantization (PQ) and Scalar Quantization (SQ) on search accuracy, query latency, memory footprint, and index construction time.
* **Clear Performance Hierarchy:** Identification of the most effective and robust algorithms, highlighting the strengths of graph-based and IVF-based methods and the limitations of LSH.
* **Actionable Recommendations:** A practical decision framework to guide the selection of the most appropriate ANN indexing strategy based on specific application constraints.

## 🛠️ Methods Evaluated

The evaluation covers nine ANN algorithms from the most influential families:

1.  **HNSW (Hierarchical Navigable Small World)**
2.  **IVF (Inverted File)**
3.  **ANNOY (Approximate Nearest Neighbors Oh Yeah)**
4.  **SCANN (Scalable Nearest Neighbors)**
5.  **Cosine LSH (Locality-Sensitive Hashing)**
6.  **HNSWPQ** (HNSW with Product Quantization)
7.  **HNSWSQ** (HNSW with Scalar Quantization)
8.  **IVFPQ** (IVF with Product Quantization)
9.  **IVFSQ** (IVF with Scalar Quantization)

## 📊 Datasets

The experiments were conducted on a diverse suite of nine real-world and semi-synthetic datasets:

* **Product Matching:** ABT-BUY, AMAZON-WALMART, AMAZON-GOOGLE, WDC
* **Bibliographic Matching:** ACM-DBLP, SCHOLAR-DBLP
* **Movies matching:** IMDB-DBPEDIA
* **Large-Scale Synthetic:** DBLP, VOTERS

All experiments were run using two different sentence-transformer models for embedding generation: `MiniLM-L6-v2` (Mini) and `Microsoft E5-large-v2` (E5).

## ⚙️ Setup and Installation

The implementations rely on several key open-source libraries. You can install them using pip:
 ```bash
    pip install -r requirements.txt
 ```

You can install a CUDA-enabled library for FAISS, be aware though of a potential incompatibility between `faiss-gpu` and `scann` concerning their required **NumPy** versions.
If the library of FAISS that supports your CUDA requires an older version of NumPy (e.g., `<2.0`), then SCANN, which may require a newer version (e.g., `>=2.0`), should be installed and run in a separate virtual environment. This will prevent package version conflicts and ensure both libraries function correctly for the experiments.

   
## ▶️ Running the Experiments

The repository is structured to allow for easy replication of the results presented in the paper.

1.  **Embedding Generation:** Use the provided scripts `embed_<DATASET>.py --model Mini` and `embed_<DATASET>.py --model E5` to generate the Mini and E5 embeddings (in terms of .pqt files) for each dataset, e.g., `embed_SCHOLAR-DBLP.py --model Mini`. For ABT-BUY and ACM-DBLP, we provide the corresponding .pqt files with the Mini embeddings.

3.  **Running a Single Experiment:** You can run the evaluation for a specific dataset using the main script. For example:
    ```bash
    python experiments.py --model Mini --dataset ABT-BUY 
    ```



