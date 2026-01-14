#!/usr/bin/env python3

"""
Generate ESM-2 embeddings.
"""

import argparse
import torch
import esm
import numpy as np
from Bio import SeqIO
import sys


def main():
    parser = argparse.ArgumentParser(description="Generate ESM-2 embeddings")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file")
    parser.add_argument("-o", "--output", required=True, help="Output .npy file")
    
    args = parser.parse_args()
    
    print("ESM-2 Embedding Generation")
    print(f"Processing: {args.input}")
    
    # Load model
    print("Loading ESM-2 model...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU")
    
    # device = torch.device("cpu")
    # model = model.to(device)
    # print("Using CPU")

    batch_converter = alphabet.get_batch_converter()
    
    # Read sequences
    sequences = []
    protein_ids = []
    
    #count = 0
    for record in SeqIO.parse(args.input, "fasta"):
        protein_ids.append(record.id)
        sequences.append(str(record.seq))
        #if(count == 10000):  break
        #count+= 1
    
    print(f"Found {len(sequences)} proteins")
    
    # Process each sequence ONE AT A TIME
    embeddings = []
    
    for i, seq in enumerate(sequences):
        # Progress update
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1}/{len(sequences)} proteins")
        
        # TRUNCATION
        if len(seq) > 1022:
            seq = seq[:1022]  # Περικοπή για να χωρέσει τα <cls>, <eos>
        
        # SINGLE sequence processing
        data = [("protein", seq)]
        labels, strs, tokens = batch_converter(data)
        
        if torch.cuda.is_available():
            tokens = tokens.cuda()
        # tokens = tokens.to(device)

        
        # INFERENCE
        with torch.no_grad():
            results = model(tokens, repr_layers=[6])
            token_embeddings = results["representations"][6]
            
            # MEAN POOLING
            protein_embedding = token_embeddings.mean(dim=1)
            
            # Convert to numpy
            embedding_np = protein_embedding.cpu().numpy()[0]
            embeddings.append(embedding_np)
    
    # Save results
    embeddings_array = np.array(embeddings)
    np.save(args.output, embeddings_array)
    
    # Save protein IDs
    ids_file = args.output.replace(".npy", "_ids.txt")
    with open(ids_file, "w") as f:
        for pid in protein_ids:
            f.write(f"{pid}\n")
    
    print(f"\n Done! Generated {len(embeddings)} embeddings")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
