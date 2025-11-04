import os
import random
from Bio import SeqIO
import numpy as np
from collections import defaultdict

# --- CONFIGURATION ---
RAW_DATA_DIR = "ncbi_raw_data"
OUTPUT_DATA_DIR = "training_data_augmented"
SPECIES_TO_PROCESS = ["human", "mouse", "zebrafish", "ecoli", "bsubtilis", "arabidopsis"]

# --- CHANGE: Using much larger chunk sizes for better context ---
# The old chunks (200-2000bp) were too small for DNABERT to find meaningful patterns.
# These larger chunks (3000-10000bp) contain more genes and regulatory elements.
CHUNK_SIZES = [3000, 4000, 5000, 6000, 8000, 10000]
SLIDING_WINDOW_STEP = 200 # A larger step is fine for these huge chunks
MAX_SAMPLES_PER_SPECIES = 500 # We'll generate fewer, but much larger, samples per species
MIN_CHUNK_LENGTH = 3000 # Discard chunks smaller than this
MAX_N_CONTENT = 0.05

def extract_chunks_from_genome_optimized(filepath, species_name):
    """Extracts large chunks from a genome file."""
    print(f"  Processing {os.path.basename(filepath)}...")
    all_chunks = []
    
    for record in SeqIO.parse(filepath, "fasta"):
        seq = str(record.seq).upper()
        seq_len = len(seq)

        for chunk_size in CHUNK_SIZES:
            if seq_len < chunk_size:
                continue
            
            # Use a sliding window to generate overlapping chunks
            for i in range(0, seq_len - chunk_size + 1, SLIDING_WINDOW_STEP):
                chunk = seq[i:i+chunk_size]
                
                # Quality Control
                if len(chunk) < MIN_CHUNK_LENGTH:
                    continue
                if chunk.count('N') / len(chunk) > MAX_N_CONTENT:
                    continue
                
                all_chunks.append(chunk)
                
                # Stop early if we have enough chunks for this file
                if len(all_chunks) >= (MAX_SAMPLES_PER_SPECIES // 5):
                    print(f"    Reached target chunk count for this file. Moving on.")
                    return all_chunks
    
    # Shuffle the chunks to ensure no order bias
    random.shuffle(all_chunks)
    return all_chunks[:MAX_SAMPLES_PER_SPECIES // 5]

def prepare_dataset():
    """Main function to prepare the augmented dataset with larger chunks."""
    print("="*60)
    print("Advanced DNA Dataset Preparation (Large Chunks)")
    print("="*60)
    
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Raw data directory '{RAW_DATA_DIR}' not found!")
        return

    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    species_chunks = defaultdict(list)

    print(f"Reading data from '{RAW_DATA_DIR}'...")
    for filename in os.listdir(RAW_DATA_DIR):
        if not filename.endswith(('.fasta', '.fa', '.fna')):
            continue
            
        filepath = os.path.join(RAW_DATA_DIR, filename)
        species_name = None
        
        for s in SPECIES_TO_PROCESS:
            if s in filename.lower():
                species_name = s
                break
        
        if not species_name:
            print(f"Warning: Could not identify species for {filename}. Skipping.")
            continue
            
        chunks = extract_chunks_from_genome_optimized(filepath, species_name)
        species_chunks[species_name].extend(chunks)
        print(f"    Extracted {len(chunks)} total chunks for {species_name} from this file.")
    
    # Balance the dataset
    print("\nBalancing the dataset...")
    final_chunks = []
    
    valid_species_chunks = {s: c for s, c in species_chunks.items() if len(c) > 0}
    if not valid_species_chunks:
        print("Error: No valid chunks were extracted. Check your input data and filters.")
        return
        
    min_samples = min(len(chunks) for chunks in valid_species_chunks.values())
    print(f"Targeting {min_samples} samples per species for balancing.")
    
    for species, chunks in valid_species_chunks.items():
        sampled_chunks = random.sample(chunks, min(len(chunks), min_samples))
        final_chunks.extend([(chunk, species) for chunk in sampled_chunks])
        print(f"  Selected {len(sampled_chunks)} samples for {species}.")

    print(f"\nSaving {len(final_chunks)} total samples to '{OUTPUT_DATA_DIR}'...")
    
    for species in SPECIES_TO_PROCESS:
        species_dir = os.path.join(OUTPUT_DATA_DIR, species)
        os.makedirs(species_dir, exist_ok=True)
    
    random.shuffle(final_chunks)
    
    for i, (chunk, species) in enumerate(final_chunks):
        output_file = os.path.join(OUTPUT_DATA_DIR, species, f"{species}_sample_{i}.fasta")
        with open(output_file, "w") as f:
            f.write(f">sample_{i}\n")
            f.write(chunk + "\n")
            
    print("\nDataset preparation complete!")
    print(f"Total samples created: {len(final_chunks)}")
    print(f"Files are saved in '{OUTPUT_DATA_DIR}' directory, ready for training.")
    print("="*60)

if __name__ == "__main__":
    random.seed(42)
    prepare_dataset()