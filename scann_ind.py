import pandas as pd
import numpy as np
import time
import scann
import os
import shutil


def check_if_match(row, gold_standard):
       id_a = row['id_A']
       if id_a not in gold_standard:
              return np.nan
       id_b = row['id_B']
       matching_ids_for_a = gold_standard.get(id_a, [])
       if id_b in matching_ids_for_a:
           return 1
       else:
           return 0



def run(df11, df22, truth, id1t, id2t):

    truthD = dict()
    a = 0

    for i, r in truth.iterrows():
        id1 = r[id1t]
        id2 = r[id2t]
        if id2 in truthD:
            ids = truthD[id2]
            ids.append(id1)
            a += 1
        else:
            truthD[id2] = [id1]
            a += 1
    matches = a
    #print("No of matches=", matches)

    # ====================================================================--

    batch_size = 10_000
    num_candidates = 5
    vectors_b = df22['v'].tolist()
    b_embeddings = np.array(vectors_b).astype(np.float32)
    d = b_embeddings.shape[1]
    vectors_a = df11['v'].tolist()
    a_embeddings = np.array(vectors_a).astype(np.float32)
    a_ids = np.array(df11['id'].tolist())
    b_ids = df22['id'].tolist()

    start_time = time.time()
    num_leaves = int(np.sqrt(len(a_embeddings)))
    num_leaves_to_search = 50
    searcher = scann.scann_ops_pybind.builder(a_embeddings, 10, "dot_product") \
       .tree(num_leaves=num_leaves, num_leaves_to_search=num_leaves_to_search, training_sample_size=25000) \
       .score_ah(2, anisotropic_quantization_threshold=0.2) \
        .reorder(100) \
        .build()
    end_time = time.time()
    index_time = round(end_time - start_time,2)


    index_directory = 'my_scann_index_dir'
    os.makedirs(index_directory, exist_ok=True)
    searcher.serialize(index_directory)

    total_size_bytes = 0
    for dirpath, dirnames, filenames in os.walk(index_directory):
       for f in filenames:
          fp = os.path.join(dirpath, f)
          total_size_bytes += os.path.getsize(fp)
    file_size_mb = total_size_bytes / (1024 * 1024)
    #print(f"ScaNN Index Memory Footprint: {file_size_mb:.2f} MB")
    shutil.rmtree(index_directory)


    tp = 0
    fp = 0
    start_time = time.time()
    all_found_pairs = []
    for i in range(0, len(b_embeddings), batch_size):
        bs = b_embeddings[i: i + batch_size]
        b_ids_in_batch = b_ids[i: i + batch_size]
        candidate_indices, distances = searcher.search_batched(bs, final_num_neighbors=num_candidates)
        flat_candidate_ids = candidate_indices.flatten()
        candidate_a_embeddings = a_embeddings[flat_candidate_ids]
        candidate_a_ids = a_ids[flat_candidate_ids]
        repeated_b_embeddings = np.repeat(bs, num_candidates, axis=0)
        repeated_b_ids = np.repeat(b_ids_in_batch, num_candidates)
        batch_pairs_df = pd.DataFrame({
          'id_A': candidate_a_ids,
          'id_B': repeated_b_ids,
        })
        all_found_pairs.append(batch_pairs_df)
    final_results_df = pd.concat(all_found_pairs, ignore_index=True)
    #print(f"Total pairs generated: {len(final_results_df)}")
    end_time = time.time()
    matching_time = round(end_time - start_time,2)
    final_results_df['is_a_match'] = final_results_df.apply(lambda row: check_if_match(row, truthD), axis=1 )
    tp = final_results_df['is_a_match'].sum()
    fp = (final_results_df['is_a_match'] == 0).sum()
    recall=round(tp / matches, 2)
    precision=round(tp / (tp + fp), 2)
    #print(f"HNSW tp={tp} fp={fp}  recall={round(tp / matches, 2)} precision={round(tp / (tp + fp), 2)} total matching time={end_time - start_time} seconds.")
    return tp, fp, recall, precision,index_time, matching_time, file_size_mb


