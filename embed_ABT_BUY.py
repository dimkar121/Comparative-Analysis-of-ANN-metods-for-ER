import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import argparse


def embed(df, text_columns, prefix, output_filename, model, name_minhash=None):
    df['combined_text'] = prefix + df[text_columns].astype(str).agg(' '.join, axis=1)
    print(df['combined_text'])

    print("Creating combined text for dense embeddings...")

    print("Generating dense embeddings... (This may take a while)")
    sentences = df['combined_text'].tolist()
    embeddings = model.encode(sentences, show_progress_bar=True)

    df['v'] = [emb.tolist() for emb in embeddings]
 

    df.drop(columns=['combined_text'], inplace=True)

    df.to_parquet(output_filename, engine="pyarrow")

    print(f"Successfully created '{output_filename}' with new feature columns.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate sentence embeddings for a paired dataset.")    
    parser.add_argument('--model', type=str, 
                        default='Mini', 
                        help='Name of the sentence-transformer model to use (e.g., "Mini", "e5").')

    args = parser.parse_args()
    print(f"Model {args.model} will be used.")
    if args.model == "Mini":      
       model_name = 'all-MiniLM-L6-v2'
       model_tag = "mini"
    else: 
       model_name = 'intfloat/e5-large-v2'
       model_tag = "e5"


    folder = "./data"
    embedding_model = SentenceTransformer(model_name)
    device_name = embedding_model.device.type
    if device_name == 'cuda':
        print(f"The model is on the GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"The model is on the CPU.")

    df = pd.read_csv("./data/Abt.csv", sep=",", encoding="unicode_escape")
    embed(
        df=df,
        text_columns=["name","description","price"],
        prefix="",
        output_filename=f'{folder}/Abt_{model_tag}.pqt',
        model=embedding_model 
    )

    df = pd.read_csv("./data/Buy.csv", sep=",", encoding="unicode_escape")
    embed(
        df=df,
        text_columns=["name","description","price"],
        prefix="",
        output_filename=f'{folder}/Buy_{model_tag}.pqt',
        model=embedding_model
    )









