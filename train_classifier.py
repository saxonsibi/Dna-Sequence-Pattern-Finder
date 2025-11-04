import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio import Entrez
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def download_more_ncbi_data():
    """Download more sequences from NCBI for each species."""
    
    Entrez.email = "your.email@example.com"  # Replace with your email
    
    # Updated species info with better search terms
    species_info = {
        'human': {
            'search_term': 'Homo sapiens[Organism] AND (chromosome[Title] OR genome[Title])',
            'max_records': 5
        },
        'mouse': {
            'search_term': 'Mus musculus[Organism] AND (mitochondrial[Title] OR "complete genome"[Title])',
            'max_records': 5
        },
        'zebrafish': {
            'search_term': 'Danio rerio[Organism] AND (mitochondrial[Title] OR "complete genome"[Title])',
            'max_records': 5
        },
        'arabidopsis': {
            'search_term': 'Arabidopsis thaliana[Organism] AND (chloroplast[Title] OR "complete genome"[Title])',
            'max_records': 5
        },
        'ecoli': {
            'search_term': 'Escherichia coli[Organism] AND (complete[Title] OR genome[Title] OR strain[Title])',
            'max_records': 5
        },
        'bsubtilis': {
            'search_term': 'Bacillus subtilis[Organism] AND (complete[Title] OR genome[Title] OR strain[Title])',
            'max_records': 5
        }
    }
    
    data_dir = 'training_data'
    
    for species, info in species_info.items():
        species_dir = os.path.join(data_dir, species)
        os.makedirs(species_dir, exist_ok=True)
        
        print(f"Downloading more {species} sequences...")
        
        try:
            # Search NCBI
            handle = Entrez.esearch(
                db="nucleotide",
                term=info['search_term'],
                retmax=info['max_records']
            )
            record = Entrez.read(handle)
            handle.close()
            
            if record['IdList']:
                # Fetch sequences
                handle = Entrez.efetch(
                    db="nucleotide",
                    id=record['IdList'],
                    rettype="fasta",
                    retmode="text"
                )
                
                sequences = list(SeqIO.parse(handle, "fasta"))
                handle.close()
                
                # Filter sequences (remove those with too many N's)
                filtered_sequences = []
                for seq_record in sequences:
                    seq_str = str(seq_record.seq).upper()
                    # More lenient filtering for problematic species
                    min_length = 500 if species in ['mouse', 'ecoli'] else 1000
                    max_n_ratio = 0.1 if species in ['mouse', 'ecoli'] else 0.05
                    
                    if len(seq_str) >= min_length and seq_str.count('N') / len(seq_str) < max_n_ratio:
                        filtered_sequences.append(seq_record)
                
                # Save to file
                if filtered_sequences:
                    output_file = os.path.join(species_dir, f"{species}_additional.fasta")
                    # Check if file already exists
                    if not os.path.exists(output_file):
                        with open(output_file, 'w') as f:
                            SeqIO.write(filtered_sequences, f, 'fasta')
                        print(f"  Downloaded {len(filtered_sequences)} sequences to {output_file}")
                    else:
                        print(f"  File already exists: {output_file}")
                else:
                    print(f"  No valid sequences found for {species}")
            else:
                print(f"  No sequences found for {species}")
                
        except Exception as e:
            print(f"  Error downloading {species}: {e}")

def generate_diverse_sequences(sequences, labels, chunk_sizes=[500, 1000, 2000], max_chunks_per_species=30):
    """
    Generate more diverse training sequences with different chunk sizes.
    """
    new_sequences = []
    new_labels = []
    
    # Track chunks per species
    chunks_per_species = Counter()
    
    for seq, label in zip(sequences, labels):
        if chunks_per_species[label] >= max_chunks_per_species:
            continue
            
        # Create chunks of different sizes
        for chunk_size in chunk_sizes:
            if len(seq) > chunk_size and chunks_per_species[label] < max_chunks_per_species:
                # Create non-overlapping chunks
                num_chunks = min(
                    len(seq) // chunk_size,
                    max_chunks_per_species - chunks_per_species[label]
                )
                
                for i in range(num_chunks):
                    start = i * chunk_size
                    end = start + chunk_size
                    chunk = seq[start:end]
                    
                    # Only add if it has enough valid bases
                    if chunk.count('N') / len(chunk) < 0.05:  # Less than 5% N's
                        new_sequences.append(chunk)
                        new_labels.append(label)
                        chunks_per_species[label] += 1
    
    return new_sequences, new_labels

def add_synthetic_variations(sequences, labels, mutation_rate=0.01, variations_per_seq=2):
    """
    Add synthetic variations to increase dataset diversity.
    """
    new_sequences = []
    new_labels = []
    
    bases = ['A', 'T', 'G', 'C']
    
    for seq, label in zip(sequences, labels):
        # Create variations of each sequence
        for _ in range(variations_per_seq):
            mutated_seq = list(seq)
            
            # Introduce random mutations
            for i in range(len(mutated_seq)):
                if np.random.random() < mutation_rate:
                    original = mutated_seq[i]
                    # Choose a different base
                    available = [b for b in bases if b != original]
                    mutated_seq[i] = np.random.choice(available)
            
            new_sequences.append(''.join(mutated_seq))
            new_labels.append(label)
    
    return new_sequences, new_labels

def prepare_training_data(data_dir, min_seq_length=500, max_sequences_per_species=100):
    """
    Prepare training data from NCBI FASTA files.
    """
    sequences = []
    labels = []
    species_list = []
    
    print("Preparing training data...")
    
    # Process each species directory
    for species_dir in os.listdir(data_dir):
        species_path = os.path.join(data_dir, species_dir)
        if not os.path.isdir(species_path):
            continue
            
        species_name = species_dir
        species_list.append(species_name)
        print(f"Processing {species_name}...")
        
        sequence_count = 0
        # Process each FASTA file in the species directory
        for filename in os.listdir(species_path):
            if not filename.endswith(('.fasta', '.fa', '.fna')):
                continue
                
            filepath = os.path.join(species_path, filename)
            try:
                for record in SeqIO.parse(filepath, "fasta"):
                    seq = str(record.seq).upper().replace('N', '')
                    
                    # More lenient length requirements for problematic species
                    min_length = 300 if species_name in ['mouse', 'ecoli'] else 500
                    
                    if len(seq) >= min_length:
                        sequences.append(seq)
                        labels.append(species_name)
                        sequence_count += 1
                        
                        if sequence_count >= max_sequences_per_species:
                            break
                
                if sequence_count >= max_sequences_per_species:
                    break
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        print(f"  Added {sequence_count} sequences for {species_name}")
    
    print(f"\nTotal sequences: {len(sequences)}")
    print(f"Species: {species_list}")
    
    # Generate more diverse sequences
    if len(sequences) < 100:
        print("\nGenerating more training data with diverse chunk sizes...")
        diverse_seqs, diverse_labels = generate_diverse_sequences(sequences, labels)
        print(f"Generated {len(diverse_seqs)} diverse chunks")
        
        # Add synthetic variations
        print("Adding synthetic variations...")
        synthetic_seqs, synthetic_labels = add_synthetic_variations(diverse_seqs, diverse_labels)
        print(f"Generated {len(synthetic_seqs)} synthetic variations")
        
        # Combine all sequences
        sequences = sequences + diverse_seqs + synthetic_seqs
        labels = labels + diverse_labels + synthetic_labels
        print(f"Total sequences after augmentation: {len(sequences)}")
    
    return sequences, labels, species_list

def extract_features(sequences, kmer_sizes=[4]):
    """
    Extract features from DNA sequences.
    """
    print("\nExtracting features...")
    
    # Extract k-mer features
    all_features = []
    vectorizers = {}
    
    for k in kmer_sizes:
        print(f"  Extracting {k}-mers...")
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k), 
                                   lowercase=False, min_df=1)
        kmer_features = vectorizer.fit_transform(sequences).toarray()
        all_features.append(kmer_features)
        vectorizers[f'kmer_{k}'] = vectorizer
        print(f"    Found {len(vectorizer.vocabulary_)} unique {k}-mers")
    
    # Extract additional features
    print("  Extracting additional features...")
    additional_features = []
    
    for seq in sequences:
        # Basic composition features
        gc_content = (seq.count('G') + seq.count('C')) / len(seq)
        at_content = (seq.count('A') + seq.count('T')) / len(seq)
        
        # CpG ratio
        cpg_count = seq.count('CG')
        c_count = seq.count('C')
        g_count = seq.count('G')
        cpg_ratio = (cpg_count * len(seq)) / (c_count * g_count) if c_count > 0 and g_count > 0 else 0
        
        # Dinucleotide frequencies
        dinucleotides = ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC', 
                         'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']
        dinuc_freqs = []
        
        for dinuc in dinucleotides:
            count = 0
            for i in range(len(seq) - 1):
                if seq[i:i+2] == dinuc:
                    count += 1
            dinuc_freqs.append(count / (len(seq) - 1) if len(seq) > 1 else 0)
        
        # Sequence complexity
        counts = Counter(seq)
        total = sum(counts.values())
        entropy = 0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        complexity = entropy / 2.0
        
        # Homopolymer runs
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
    
    # Combine all features
    X = np.hstack(all_features + [np.array(additional_features)])
    
    print(f"  Total features: {X.shape[1]}")
    
    return X, vectorizers

def train_classifier(X, y, species_list):
    """
    Train a Random Forest classifier with PCA for dimensionality reduction.
    """
    print("\nTraining classifier...")
    
    # Check if we have enough samples per class
    label_counts = Counter(y)
    min_samples = min(label_counts.values())
    
    print(f"Minimum samples per class: {min_samples}")
    
    # Use appropriate splitting strategy
    if min_samples >= 2:
        print("Using stratified train/test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        print("Using simple train/test split (no stratification)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create pipeline with PCA for dimensionality reduction
    n_components = min(50, X.shape[1] // 2)  # Use half of features or 50, whichever is smaller
    
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
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Cross-validation
    if min_samples >= 3 and len(X_train) >= 10:
        cv_folds = min(5, min_samples)
        cv_scores = cross_val_score(model, X, y, cv=cv_folds)
        print(f"\nCross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    else:
        print("\nSkipping cross-validation due to insufficient samples per class.")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=species_list)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=species_list, yticklabels=species_list, 
                cmap='Blues', cbar=False)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, {
        'accuracy': accuracy,
        'confusion_matrix': cm
    }

def save_model(model, vectorizers, species_list, model_dir='model'):
    """Save the trained model and related objects."""
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'dna_classifier.pkl'))
    joblib.dump(vectorizers, os.path.join(model_dir, 'vectorizers.pkl'))
    joblib.dump(species_list, os.path.join(model_dir, 'species_list.pkl'))
    
    # Save feature names for reference
    feature_names = []
    for k in [4]:  # Only 4-mers in this version
        feature_names.extend([f'4-mer_{km}' for km in vectorizers[f'kmer_{k}'].get_feature_names_out()])
    feature_names.extend(['GC_content', 'AT_content', 'CpG_ratio', 'Complexity'])
    feature_names.extend([f'Dinuc_{dinuc}' for dinuc in ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC', 
                                                          'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']])
    feature_names.extend([f'Longest_{base}' for base in 'ATCG'])
    
    joblib.dump(feature_names, os.path.join(model_dir, 'feature_names.pkl'))
    
    print(f"\nModel saved to {model_dir}/")
    print("Files saved:")
    print("  - dna_classifier.pkl")
    print("  - vectorizers.pkl")
    print("  - species_list.pkl")
    print("  - feature_names.pkl")

def main():
    """Main training function."""
    print("=" * 60)
    print("Enhanced DNA Sequence Classifier Training")
    print("=" * 60)
    
    data_dir = 'training_data'
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        return
    
    # Step 1: Download more data from NCBI
    print("Step 1: Downloading additional data from NCBI...")
    download_more_ncbi_data()
    
    # Step 2: Prepare training data
    print("\nStep 2: Preparing training data...")
    sequences, labels, species_list = prepare_training_data(data_dir)
    
    if len(sequences) == 0:
        print("Error: No sequences found! Please check your data directory.")
        return
    
    print(f"\nDataset summary:")
    for species in species_list:
        count = labels.count(species)
        print(f"  {species}: {count} sequences")
    
    # Step 3: Extract features
    print("\nStep 3: Extracting features...")
    X, vectorizers = extract_features(sequences)
    
    # Step 4: Train classifier
    print("\nStep 4: Training classifier...")
    model, metrics = train_classifier(X, labels, species_list)
    
    # Step 5: Save model
    print("\nStep 5: Saving model...")
    save_model(model, vectorizers, species_list)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Final model accuracy: {metrics['accuracy']:.4f}")
    print("\nYou can now use this model in your Flask application!")

if __name__ == "__main__":
    main()