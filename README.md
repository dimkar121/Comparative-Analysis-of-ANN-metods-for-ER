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

    **`requirements.txt`:**
    ```
    pandas
    numpy
    scikit-learn
    faiss-gpu # or faiss-cpu if you don't have a GPU
    annoy
    scann
    lshashpy3
    sentence-transformers    
    ```

## ▶️ Running the Experiments

The repository is structured to allow for easy replication of the results presented in the paper.

1.  **Data Preparation:** Ensure the datasets are placed in the appropriate `/data` directory. Scripts for downloading and preprocessing the datasets are provided.

2.  **Embedding Generation:** Use the provided scripts to generate the Mini and E5 embeddings for each dataset.

3.  **Running a Single Experiment:** You can run the evaluation for a specific ANN method on a given dataset using the main script. For example:
    ```bash
    python run_experiment.py --method HNSW --dataset WDC --embedding_model E5
    ```

4.  **Reproducing All Results:** A shell script is provided to run all experiments and generate the summary CSV files.
    ```bash
    bash run_all.sh
    ```

## 📈 Results

The detailed results, including all performance metrics for each method on every dataset, are available in the `/results` directory. The paper provides a comprehensive analysis and discussion of these findings.

