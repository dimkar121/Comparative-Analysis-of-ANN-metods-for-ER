import pandas as pd
import numpy as np
from lshashpy3 import LSHash
from tqdm import tqdm
import time
import pickle
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

    # Normalize vectors for cosine similarity (angular distance in LSH)
    a_embeddings /= np.linalg.norm(a_embeddings, axis=1, keepdims=True)
    b_embeddings /= np.linalg.norm(b_embeddings, axis=1, keepdims=True)
    embedding_dim = a_embeddings.shape[1]
    k=15
    L=60
    if d > 364:
        k = 25
        L = 70
    lsh = LSHash(hash_size=k,  num_hashtables=L, input_dim=embedding_dim)

    # Index each vector along with its original ID
    start_time = time.time()
    for i, vec in enumerate(a_embeddings):
        a_id = a_ids[i]
        lsh.index(vec, extra_data=a_id)
    end_time = time.time()
    #print(f"Index built in {end_time - start_time:.2f} seconds.")
    index_time = round(end_time-start_time, 2)

    # --- Measure the Memory Footprint ---
    index_filename = 'my_lsh_index.pkl'
    with open(index_filename, 'wb') as f:
        pickle.dump(lsh, f)
    file_size_bytes = os.path.getsize(index_filename)
    file_size_mb = file_size_bytes / (1024 * 1024)
    #print(f"LSHash Index Memory Footprint: {file_size_mb:.2f} MB")
    # Clean up the file
    os.remove(index_filename)



    tp = 0
    fp = 0
    k_neighbors = 5
    start_time = time.time()
    all_found_pairs=[]
    for i in tqdm(range(b_embeddings.shape[0]), desc="Evaluating queries"):
        b_id = b_ids[i]
        # Query the LSH index. It returns a list of tuples: ((vector, id), distance)
        response = lsh.query(b_embeddings[i], num_results=k_neighbors, distance_func="cosine")
        a_ids_ = [item[0][1] for item in response]
        b_ids_=np.repeat(b_id, len(a_ids_))
        # Extract just the IDs of the retrieved neighbors
        batch_pairs_df = pd.DataFrame({
          'id_A': a_ids_,
           #'title_A': titles1,
          'id_B': b_ids_,
          #'title_B': titles2
        })
        all_found_pairs.append(batch_pairs_df)
    end_time = time.time()
    final_results_df = pd.concat(all_found_pairs, ignore_index=True)
    #print(f"Total pairs generated: {len(final_results_df)}")
    matching_time = round(end_time-start_time, 2)
    # Apply the function to create a new 'is_a_match' column
    final_results_df['is_a_match'] = final_results_df.apply(lambda row: check_if_match(row, truthD), axis=1 )

    tp = final_results_df['is_a_match'].sum()
    fp = (final_results_df['is_a_match'] == 0).sum()
    recall=round(tp / matches, 2)
    precision=round(tp / (tp + fp), 2)
    #print(f"HNSW tp={tp} fp={fp}  recall={round(tp / matches, 2)} precision={round(tp / (tp + fp), 2)} total matching time={end_time - start_time} seconds.")
    return tp, fp, recall, precision,index_time, matching_time,  file_size_mb 
 



