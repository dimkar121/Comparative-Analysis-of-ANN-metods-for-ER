import pandas as pd
import numpy as np
from annoy import AnnoyIndex
import time
from tqdm import tqdm
import os

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

    vectors_b = df22['v'].tolist()
    b_embeddings = np.array(vectors_b).astype(np.float32)
    d = b_embeddings.shape[1]
    vectors_a = df11['v'].tolist()
    a_embeddings = np.array(vectors_a).astype(np.float32)
    a_ids = np.array(df11['id'].tolist())
    b_ids = df22['id'].tolist()

    a_embeddings /= np.linalg.norm(a_embeddings, axis=1, keepdims=True)
    b_embeddings /= np.linalg.norm(b_embeddings, axis=1, keepdims=True)
    embedding_dim = a_embeddings.shape[1]

    annoy_index = AnnoyIndex(embedding_dim, 'angular')

    for i in tqdm(range(a_embeddings.shape[0])):
         annoy_index.add_item(i, a_embeddings[i])

    start_time = time.time()
    n_trees=150
    annoy_index.build(n_trees)
    end_time = time.time()
    #print(f"Index built in {end_time - start_time:.2f} seconds.")
    index_time = round(end_time - start_time,2)


    index_filename = 'my_annoy_index.ann'
    annoy_index.save(index_filename)
    file_size_bytes = os.path.getsize(index_filename)
    file_size_mb = file_size_bytes / (1024 * 1024)
    #print(f"Annoy Index Memory Footprint: {file_size_mb:.2f} MB")
    # Clean up the file
    os.remove(index_filename)



    k_neighbors = 5 # The number of nearest neighbors to find

    all_neighbors = []
    all_distances = []
    tp=0
    fp=0
    total_candidates_generated = 0
    total_positives_in_gold_standard = sum(len(v) for v in truthD.values())
    all_found_pairs = []
    start_time = time.time()
    for i in range(b_embeddings.shape[0]):
      neighbors, distances = annoy_index.get_nns_by_vector(
        b_embeddings[i],
        k_neighbors,
        search_k=n_trees * k_neighbors,
        include_distances=True
      )
      total_candidates_generated += len(neighbors)
      b_id = b_ids[i]
      a_ids_ = []
      b_ids_=np.repeat(b_id, len(neighbors))
      for neighbor_idx in neighbors:
         a_id = a_ids[neighbor_idx]
         a_ids_.append(a_id)
      batch_pairs_df = pd.DataFrame({
          'id_A': a_ids_,
          'id_B': b_ids_,
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
