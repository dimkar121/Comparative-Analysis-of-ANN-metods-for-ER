import pandas as pd
import numpy as  np
import hnsw
import hnswpq
import hnswsq
import ivf
import ivfpq
import ivfsq
import annoy_ind
import lsh
import scann_ind
import argparse

if __name__ == '__main__':
     parser = argparse.ArgumentParser(description="Generate sentence embeddings for a paired dataset.")
     parser.add_argument('--model', type=str,
                          default='Mini',
                          help='Name of the sentence-transformer model to use ("Mini", "e5").')
     parser.add_argument('--dataset', type=str,
                            default='SCHOLAR-DBLP',
                            help='Name of the paired dataset to use (e.g., "SCHOLAR-DBLP", "AMAZON-WALMART").')
     

 
     args = parser.parse_args()
     print(f"Embedding Model={args.model} will be used on dataset={args.dataset}.")
     if args.model == "Mini":
         model = "mini"
     elif args.model == "E5":
         model = "e5"

     dataset = args.dataset 

   
     if dataset ==  "IMDB-DBPEDIA":
          df11 = pd.read_parquet(f"./data/imdb_{model}.pqt")
          df22 = pd.read_parquet(f"./data/dbpedia_{model}.pqt")
          df11['id'] = pd.to_numeric(df11['id'], errors='coerce')
          df22['id'] = pd.to_numeric(df22['id'], errors='coerce')
          truth = pd.read_csv("./data/truth_imdb_dbpedia.csv", sep="|", encoding="utf-8", keep_default_na=False)
          valid_d1_ids = set(df11['id'].values)
          valid_d2_ids = set(df22['id'].values)
          mask_to_keep = truth['D1'].isin(valid_d1_ids) & truth['D2'].isin(valid_d2_ids)
          truth = truth[mask_to_keep].copy()
          id1t = "D2"
          id2t =  "D1"
     elif dataset == "AMAZON-WALMART":
          truth_file="./data/truth_amazon_walmart.tsv"
          df22 = pd.read_parquet(f"./data/walmart_products_{model}.pqt")
          df11 = pd.read_parquet(f"./data/amazon_products_{model}.pqt")
          if model == "e5":
            df22["id"] = df22["id"].astype(str)
            df11["id"] = df11["id"].astype(str) 
          truth = pd.read_csv("./data/truth_amazon_walmart.tsv", sep="\t", encoding="unicode_escape", keep_default_na=False)
          if model=="e5":
            truth['id1'] = truth['id1'].astype(str)
            truth['id2'] = truth['id2'].astype(str)
          id1t = "id2"
          id2t = "id1"
     elif dataset == "ACM-DBLP":
         truth_file="./data/truth_ACM_DBLP.csv"
         truth = pd.read_csv(truth_file, sep=",", encoding="utf-8", keep_default_na=False)
         df22 = pd.read_parquet(f"./data/ACM_{model}.pqt")
         df11 = pd.read_parquet(f"./data/DBLP_{model}.pqt")
         if model == "e5":
            df22["id"] = df22["id"].astype(int)
         id1t = "idACM"
         id2t = "idDBLP"
     elif dataset == "ABT-BUY":    
         truth_file="./data/truth_abt_buy.csv"
         truth = pd.read_csv(truth_file, sep=",", encoding="utf-8", keep_default_na=False)
         df22 = pd.read_parquet(f"./data/Abt_{model}.pqt")
         df11 = pd.read_parquet(f"./data/Buy_{model}.pqt")
         id1t = "idAbt"
         id2t = "idBuy"
     elif dataset =="AMAZON-GOOGLE":    
         truth_file="./data/truth_Amazon_googleProducts.csv"
         truth = pd.read_csv(truth_file, sep=",", encoding="utf-8", keep_default_na=False)
         df22 = pd.read_parquet(f"./data/Amazon_{model}.pqt")
         df11 = pd.read_parquet(f"./data/Google_{model}.pqt")
         id1t = "idAmazon"
         id2t = "idGoogleBase"
     elif dataset == "SCHOLAR-DBLP":
         truth_file="./data/truth_Scholar_DBLP.csv"
         truth = pd.read_csv(truth_file, sep=",", encoding="utf-8", keep_default_na=False)
         df22 = pd.read_parquet(f"./data/DBLP2_{model}.pqt")
         df11 = pd.read_parquet(f"./data/Scholar_{model}.pqt")
         id1t = "idDBLP"
         id2t = "idScholar"
     elif dataset == "WDC":
         truth_file="./data/truth_WDC.csv"
         truth = pd.read_csv(truth_file, sep=",", encoding="utf-8", keep_default_na=False)
         df22 = pd.read_parquet(f"./data/WDCA_{model}.pqt")
         df11 = pd.read_parquet(f"./data/WDCB_{model}.pqt")
         df11["id"] = df11["id"].astype('int64')
         df22["id"] = df22["id"].astype('int64')
         id2t = "id_B"
         id1t = "id_A"
     elif dataset == "DBLP":
         truth_file="./data/truth_DBLP.csv"
         truth = pd.read_csv(truth_file, sep=",", encoding="utf-8", keep_default_na=False)
         df22 = pd.read_parquet(f"./data/test_dblp_A_{model}.pqt")
         df11 = pd.read_parquet(f"./data/test_dblp_B_{model}.pqt")
         df11["id"] = df11["id"].astype('int64')
         df22["id"] = df22["id"].astype('int64')
         id2t = "id1"
         id1t = "id2"
     elif dataset == "VOTERS":
         truth_file="./data/truth_VOTERS.csv"
         truth = pd.read_csv(truth_file, sep=",", encoding="utf-8", keep_default_na=False)
         df22 = pd.read_parquet(f"./data/test_voters_A_{model}.pqt")
         df11 = pd.read_parquet(f"./data/test_voters_B_{model}.pqt")
         id2t = "id1"
         id1t = "id2"




     methods = ['HNSW', 'HNSWPQ','HNSWSQ', 'IVF', 'IVFPQ', 'IVFSQ', 'ANNOY','LSH', 'SCANN']
     indexes = [hnsw, hnswpq, hnswsq, ivf, ivfpq, ivfsq, annoy_ind,lsh, scann_ind ]
     recall_arr = []
     precision_arr = []
     index_time_arr = []
     match_time_arr = []
     mem_arr = []
     for ind, m in zip(indexes, methods):     
          tp, fp, recall, precision, index_time, match_time,mem =  ind.run(truth=truth,
                   id1t = id1t,
                   id2t =  id2t,
                   df11=df11,
                   df22=df22 )
          print(f"{m} tp={tp} fp={fp}  recall={recall} precision={precision} indexing time={index_time} secs  matching time={match_time} secs memory footprint={mem} MB.")
          recall_arr.append(recall)
          precision_arr.append(precision)
          index_time_arr.append(index_time)
          match_time_arr.append(match_time)
          mem_arr.append(mem)  
      
      
     results = {          
          'ANN method': methods,
          'recall': recall_arr,
          'precision': precision_arr,
          'index_time': index_time_arr,
          'match_time': match_time_arr,
          'memory_mb':mem_arr
     }
          
     df = pd.DataFrame(results)
     df['f1_score'] = 2 * (df['precision'] * df['recall']) / (df['precision'] + df['recall'])
     df['f1_score'] = df['f1_score'].round(2)
     df['memory_mb'] = df['memory_mb'].round(2)
     csv_output_path = f'./summary_{dataset}_{model}.csv'
     df.to_csv(csv_output_path, index=False)
  

