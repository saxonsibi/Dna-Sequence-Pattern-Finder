import os
import joblib
import numpy as np
from Bio import SeqIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from collections import Counter

def recreate_and_save_model():
    """Recreate the model from training data and save it."""
    
    print("Recreating and saving the model...")
    
    # Load training data
    data_dir = 'training_data'
    sequences = []
    labels = []
    species_list = []
    
    print("Loading training data...")
    
    for species_dir in os.listdir(data_dir):
        species_path = os.path.join(data_dir, species_dir)
        if not os.path.isdir(species_path):
            continue
            
        species_name = species_dir
        species_list.append(species_name)
        print(f"  Loading {species_name}...")
        
        for filename in os.listdir(species_path):
            if not filename.endswith(('.fasta', '.fa', '.fna')):
                continue
                
            filepath = os.path.join(species_path, filename)
            try:
                for record in SeqIO.parse(filepath, "fasta"):
                    seq = str(record.seq).upper().replace('N', '')
                    if len(seq) >= 500:
                        sequences.append(seq)
                        labels.append(species_name)
            except Exception as e:
                print(f"    Error reading {filename}: {e}")
    
    print(f"\nLoaded {len(sequences)} sequences from {len(species_list)} species")
    
    # Generate more data (same as in training script)
    print("\nGenerating more training data...")
    
    # Diverse chunks
    new_sequences = []
    new_labels = []
    chunks_per_species = Counter()
    
    for seq, label in zip(sequences, labels):
        for chunk_size in [500, 1000, 2000]:
            if len(seq) > chunk_size and chunks_per_species[label] < 30:
                num_chunks = min(len(seq) // chunk_size, 30 - chunks_per_species[label])
                for i in range(num_chunks):
                    start = i * chunk_size
                    end = start + chunk_size
                    chunk = seq[start:end]
                    if chunk.count('N') / len(chunk) < 0.05:
                        new_sequences.append(chunk)
                        new_labels.append(label)
                        chunks_per_species[label] += 1
    
    # Synthetic variations
    synthetic_sequences = []
    synthetic_labels = []
    bases = ['A', 'T', 'G', 'C']
    
    for seq, label in zip(new_sequences, new_labels):
        for _ in range(2):
            mutated_seq = list(seq)
            for i in range(len(mutated_seq)):
                if np.random.random() < 0.01:
                    original = mutated_seq[i]
                    available = [b for b in bases if b != original]
                    mutated_seq[i] = np.random.choice(available)
            synthetic_sequences.append(''.join(mutated_seq))
            synthetic_labels.append(label)
    
    # Combine all sequences
    all_sequences = sequences + new_sequences + synthetic_sequences
    all_labels = labels + new_labels + synthetic_labels
    
    print(f"Total sequences after augmentation: {len(all_sequences)}")
    
    # Extract features
    print("\nExtracting features...")
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(4, 4), 
                               lowercase=False, min_df=1)
    X_kmers = vectorizer.fit_transform(all_sequences).toarray()
    
    additional_features = []
    for seq in all_sequences:
        gc_content = (seq.count('G') + seq.count('C')) / len(seq)
        at_content = (seq.count('A') + seq.count('T')) / len(seq)
        cpg_count = seq.count('CG')
        c_count = seq.count('C')
        g_count = seq.count('G')
        cpg_ratio = (cpg_count * len(seq)) / (c_count * g_count) if c_count > 0 and g_count > 0 else 0
        
        dinucleotides = ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC', 
                         'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']
        dinuc_freqs = []
        for dinuc in dinucleotides:
            count = 0
            for i in range(len(seq) - 1):
                if seq[i:i+2] == dinuc:
                    count += 1
            dinuc_freqs.append(count / (len(seq) - 1) if len(seq) > 1 else 0)
        
        counts = Counter(seq)
        total = sum(counts.values())
        entropy = 0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        complexity = entropy / 2.0
        
        longest_runs = []
        for base in 'ATCG':
            max_run = 0
            current_run = 0
            for b in seq:
                if b == base:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            longest_runs.append(max_run)
        
        additional_features.append([
            gc_content, at_content, cpg_ratio, complexity
        ] + dinuc_freqs + longest_runs)
    
    X = np.hstack([X_kmers, np.array(additional_features)])
    vectorizers = {'kmer_4': vectorizer}
    
    print(f"Extracted {X.shape[1]} features")
    
    # Train model
    print("\nTraining model...")
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    n_components = min(50, X.shape[1] // 2)
    model = Pipeline([
        ('pca', PCA(n_components=n_components, random_state=42)),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate
    from sklearn.metrics import accuracy_score
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save model
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'dna_classifier.pkl'))
    joblib.dump(vectorizers, os.path.join(model_dir, 'vectorizers.pkl'))
    joblib.dump(species_list, os.path.join(model_dir, 'species_list.pkl'))
    
    # Save feature names
    feature_names = []
    feature_names.extend([f'4-mer_{km}' for km in vectorizers['kmer_4'].get_feature_names_out()])
    feature_names.extend(['GC_content', 'AT_content', 'CpG_ratio', 'Complexity'])
    feature_names.extend([f'Dinuc_{dinuc}' for dinuc in ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC', 
                                                      'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']])
    feature_names.extend([f'Longest_{base}' for base in 'ATCG'])
    
    joblib.dump(feature_names, os.path.join(model_dir, 'feature_names.pkl'))
    
    print(f"\nModel saved to {model_dir}/")
    print("Files created:")
    print("  - dna_classifier.pkl")
    print("  - vectorizers.pkl")
    print("  - species_list.pkl")
    print("  - feature_names.pkl")
    
    # Verify files were created
    files = os.listdir(model_dir)
    print(f"\nFiles in model directory: {files}")
    
    # Test loading
    try:
        test_model = joblib.load(os.path.join(model_dir, 'dna_classifier.pkl'))
        test_species = joblib.load(os.path.join(model_dir, 'species_list.pkl'))
        print(f"\nModel loaded successfully! Can predict: {test_species}")
        return True
    except Exception as e:
        print(f"\nError loading model: {e}")
        return False

if __name__ == "__main__":
    recreate_and_save_model()