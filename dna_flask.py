import os
import numpy as np
import pandas as pd
from Bio import SeqIO
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
import traceback
import gc
from functools import wraps
import secrets
from pathlib import Path
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Union
import re
from collections import Counter, defaultdict
import math
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from sklearn.pipeline import Pipeline # ADDED: For type checking
# Add these imports at the top of your app.py
from flask import request, jsonify
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
class Config:
    SECRET_KEY = secrets.token_hex(32)  # Generate a secure random key
    UPLOAD_FOLDER = 'uploads'
    REFERENCES_FOLDER = 'references'  # New folder for reference sequences
    MODEL_DIR = 'model'  # Directory for trained model
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2GB
    MAX_SEQUENCES = 5000  # Limit for memory efficiency
    SESSION_TIMEOUT = 3600  # 1 hour
    ALLOWED_EXTENSIONS = {'fasta', 'fa', 'txt'}
    KMER_SIZE = 6  # For classification analysis
    SAMPLE_SIZE = 2000000  # For large sequences
    # NEW: Added GC content distribution parameters
    GC_WINDOW_SIZE = 1000  # Window size for GC content distribution
    GC_WINDOW_STEP = 500   # Step size for sliding window

app.config.from_object(Config)

# Ensure upload, reference, and model folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REFERENCES_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_DIR'], exist_ok=True)

# Load the trained model and related objects
try:
    model = joblib.load(os.path.join(app.config['MODEL_DIR'], 'dna_classifier.pkl'))
    vectorizers = joblib.load(os.path.join(app.config['MODEL_DIR'], 'vectorizers.pkl'))
    species_list = joblib.load(os.path.join(app.config['MODEL_DIR'], 'species_list.pkl'))
    feature_names = joblib.load(os.path.join(app.config['MODEL_DIR'], 'feature_names.pkl'))
    logger.info("Model loaded successfully!")
    
    # ADDED: Check if model is a Pipeline and log its steps for better debugging
    if isinstance(model, Pipeline):
        logger.info(f"Model is a Pipeline with steps: {list(model.named_steps.keys())}")
    else:
        logger.warning("Model is not a Pipeline. This might indicate a version mismatch or an issue with the saved model.")

    logger.info(f"Available species for classification: {species_list}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    vectorizers = None
    species_list = None
    feature_names = None

# Create a dictionary to store file metadata
file_metadata = {}

# Reference sequence metadata storage
reference_metadata = {}

# Session timeout decorator
def session_timeout_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'last_activity' in session:
            last_activity = session['last_activity']
            if time.time() - last_activity > app.config['SESSION_TIMEOUT']:
                session.clear()
                flash('Your session has expired. Please upload your file again.', 'info')
                return redirect(url_for('upload'))
        session['last_activity'] = time.time()
        return f(*args, **kwargs)
    return decorated_function

# Add error handler for large files
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    flash('File is too large for analysis. Please use a file smaller than 2GB.', 'danger')
    return redirect(url_for('upload'))

# Add error handler for 404
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# Add error handler for 500
@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500

def allowed_file(filename: str) -> bool:
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def calculate_file_hash(filepath: str) -> str:
    """Calculate SHA256 hash of a file for integrity checking."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def cleanup_old_files():
    """Remove files older than 24 hours from the upload folder."""
    current_time = time.time()
    for folder in [app.config['UPLOAD_FOLDER'], app.config['REFERENCES_FOLDER']]:
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > 86400:  # 24 hours in seconds
                    try:
                        os.remove(filepath)
                        logger.info(f"Removed old file: {filename}")
                    except Exception as e:
                        logger.error(f"Error removing file {filename}: {str(e)}")

def get_reference_files():
    """Get list of reference files with metadata."""
    if not os.path.exists(app.config['REFERENCES_FOLDER']):
        return []
    
    references = []
    for f in os.listdir(app.config['REFERENCES_FOLDER']):
        filepath = os.path.join(app.config['REFERENCES_FOLDER'], f)
        if os.path.isfile(filepath) and allowed_file(f):
            references.append({
                'filename': f,
                'species': identify_species(f),
                'size': os.path.getsize(filepath),
                'upload_time': reference_metadata.get(f, {}).get('upload_time', 'Unknown')
            })
    
    return references

def identify_species(filename):
    """Identify species based on filename."""
    filename_lower = filename.lower()
    
    if 'ecoli' in filename_lower:
        return 'E. coli bacteria'
    elif 'human' in filename_lower:
        return 'Human chromosome 1'
    elif 'drosophila' in filename_lower:
        return 'Fruit fly'
    elif 'yeast' in filename_lower:
        return 'Yeast'
    elif 'zebrafish' in filename_lower:
        return 'Zebrafish'
    elif 'mouse' in filename_lower:
        return 'Mouse'
    elif 'arabidopsis' in filename_lower:
        return 'Arabidopsis thaliana (Plant)'
    elif 'bsubtilis' in filename_lower:
        return 'B. subtilis'
    else:
        return 'Unknown'

def check_model_compatibility():
    """
    Check if the model is compatible with current feature extraction.
    """
    if model is None:
        logger.error("Model not loaded")
        return False
    
    try:
        # Get the expected number of features from the vectorizer
        kmer_vectorizer = vectorizers.get('kmer_4')
        if kmer_vectorizer is None:
            logger.error("kmer_4 vectorizer not found")
            return False
        
        kmer_features = len(kmer_vectorizer.vocabulary_)
        additional_features = 4 + 16 + 4  # gc, at, cpg, complexity + 16 dinucleotides + 4 homopolymer runs
        total_features = kmer_features + additional_features
        
        logger.info(f"Expected features: {total_features} (kmer: {kmer_features}, additional: {additional_features})")
        
        # Check if model expects this many features
        if hasattr(model, 'n_features_in_'):
            logger.info(f"Model expects {model.n_features_in_} features")
            return model.n_features_in_ == total_features
        
        return True
    except Exception as e:
        logger.error(f"Error checking model compatibility: {e}")
        return False

def extract_features_single(sequence):
    """
    Extract features from a single DNA sequence, matching the training process.
    NOTE: This function must be IDENTICAL to the one used in training.
    """
    # Ensure sequence is uppercase and remove N's
    seq = str(sequence).upper().replace('N', '')
    
    # Check if sequence is too short
    if len(seq) < 500:
        logger.warning(f"Sequence too short for reliable classification: {len(seq)} bp")
        return None
    
    # Extract k-mer features - this must match training exactly
    all_features = []
    
    # Only use 4-mers as in training
    for k in [4]:
        vectorizer = vectorizers[f'kmer_{k}']
        kmer_features = vectorizer.transform([seq]).toarray()
        all_features.append(kmer_features)
    
    # Extract additional features - must match training exactly
    gc_content = (seq.count('G') + seq.count('C')) / len(seq) if len(seq) > 0 else 0
    at_content = (seq.count('A') + seq.count('T')) / len(seq) if len(seq) > 0 else 0
    
    # CpG ratio
    cpg_count = seq.count('CG')
    c_count = seq.count('C')
    g_count = seq.count('G')
    cpg_ratio = (cpg_count * len(seq)) / (c_count * g_count) if c_count > 0 and g_count > 0 and len(seq) > 0 else 0
    
    # Dinucleotide frequencies - must match training exactly
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
        p = count / total if total > 0 else 0
        if p > 0:
            entropy -= p * math.log2(p)
    complexity = entropy / 2.0 if total > 0 else 0
    
    # Homopolymer runs - must match training exactly
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
    
    # Combine all features in the same order as training
    additional_features = [
        gc_content, at_content, cpg_ratio, complexity
    ] + dinuc_freqs + longest_runs
    
    # Combine all features
    X = np.hstack(all_features + [np.array(additional_features).reshape(1, -1)])
    
    return X

def predict_sequence(sequence):
    """
    Predict the species for a given DNA sequence.
    """
    if model is None:
        logger.error("Model not available for classification")
        return None, None
    
    try:
        # Extract features
        features = extract_features_single(sequence)
        
        if features is None:
            logger.error("Feature extraction failed")
            return None, None
        
        # Debug: Print feature shape
        logger.info(f"Extracted features shape: {features.shape}")
        
        # Make prediction. The Pipeline model will automatically apply PCA transformation.
        prediction = model.predict(features)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(features)[0]
        
        # Create a dictionary of species probabilities
        species_probs = {species: prob for species, prob in zip(species_list, probabilities)}
        
        # ADDED: Log all probabilities for better diagnostics
        sorted_probs = sorted(species_probs.items(), key=lambda item: item[1], reverse=True)
        logger.info(f"Prediction successful: {prediction} with confidence {species_probs[prediction]:.3f}")
        logger.info(f"Full probability distribution: {sorted_probs}")

        return prediction, species_probs
    except Exception as e:
        logger.error(f"Error predicting sequence: {e}")
        logger.error(traceback.format_exc())
        return None, None

# NEW: Function to calculate GC content in sliding windows
def calculate_gc_distribution(sequence, window_size=1000, step_size=500):
    """
    Calculate GC content in sliding windows across the sequence.
    Returns positions and GC content values for plotting.
    """
    logger.info(f"Calculating GC content distribution with window size {window_size} and step {step_size}")
    
    try:
        seq_length = len(sequence)
        positions = []
        gc_values = []
        
        # Use numpy for faster array operations
        seq_array = np.array(list(sequence))
        
        for i in range(0, seq_length - window_size + 1, step_size):
            window = seq_array[i:i+window_size]
            g_count = np.sum(window == 'G')
            c_count = np.sum(window == 'C')
            gc_content = (g_count + c_count) / window_size if window_size > 0 else 0
            
            positions.append(i + window_size // 2)  # Center position of window
            gc_values.append(gc_content * 100)  # Convert to percentage
        
        # Handle the last window if it doesn't fit perfectly
        if seq_length % step_size != 0 and seq_length > window_size:
            i = seq_length - window_size
            window = seq_array[i:i+window_size]
            g_count = np.sum(window == 'G')
            c_count = np.sum(window == 'C')
            gc_content = (g_count + c_count) / window_size if window_size > 0 else 0
            
            positions.append(i + window_size // 2)
            gc_values.append(gc_content * 100)
        
        logger.info(f"GC content distribution calculated for {len(positions)} windows")
        return positions, gc_values
    
    except Exception as e:
        logger.error(f"Error calculating GC distribution: {str(e)}")
        return [], []

# Routes
@app.route('/')
def index():
    """Home page."""
    return render_template('index.html', species_list=species_list)

@app.route('/upload')
def upload():
    """File upload page."""
    # Get available reference sequences for the form
    references = get_reference_files()
    return render_template('upload.html', references=references)

@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict species for a DNA sequence."""
    if model is None:
        flash('Model not loaded. Please check the server logs.', 'error')
        return redirect(url_for('index'))
    
    sequence = request.form.get('sequence', '').strip()
    
    if not sequence:
        flash('Please enter a DNA sequence.', 'error')
        return redirect(url_for('index'))
    
    # Make prediction
    prediction, species_probs = predict_sequence(sequence)
    
    if prediction is None:
        flash('Error making prediction. Please check your sequence.', 'error')
        return redirect(url_for('index'))
    
    # Calculate sequence statistics
    seq_stats = {
        'length': len(sequence),
        'gc_content': (sequence.count('G') + sequence.count('C')) / len(sequence) * 100 if len(sequence) > 0 else 0,
        'at_content': (sequence.count('A') + sequence.count('T')) / len(sequence) * 100 if len(sequence) > 0 else 0,
        'n_content': sequence.count('N') / len(sequence) * 100 if len(sequence) > 0 else 0
    }
    
    return render_template('results.html', 
                          sequence=sequence,
                          prediction=prediction,
                          species_probs=species_probs,
                          seq_stats=seq_stats)

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict species for multiple DNA sequences from a file."""
    if model is None:
        flash('Model not loaded. Please check the server logs.', 'error')
        return redirect(url_for('upload'))
    
    # Check if the post request has the file part
    if 'dna_file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('upload'))
    
    file = request.files['dna_file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('upload'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Parse the FASTA file
            sequences = []
            sequence_ids = []
            
            for record in SeqIO.parse(filepath, "fasta"):
                sequences.append(str(record.seq))
                sequence_ids.append(record.id)
            
            if not sequences:
                flash('No valid sequences found in the file.', 'error')
                return redirect(url_for('upload'))
            
            # Make predictions for all sequences
            results = []
            for seq_id, sequence in zip(sequence_ids, sequences):
                prediction, species_probs = predict_sequence(sequence)
                
                if prediction is not None:
                    # Get the top 3 predictions
                    sorted_species = sorted(species_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    results.append({
                        'id': seq_id,
                        'sequence': sequence[:100] + '...' if len(sequence) > 100 else sequence,
                        'prediction': prediction,
                        'confidence': species_probs[prediction],
                        'top_predictions': sorted_species
                    })
            
            return render_template('batch_results.html', results=results)
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('upload'))
    else:
        flash('Invalid file type. Please upload a FASTA file (.fasta, .fa, .fna).', 'error')
        return redirect(url_for('upload'))

# Reference sequence management routes
@app.route('/references')
def list_references():
    """List all reference sequences."""
    references = get_reference_files()
    return render_template('references.html', references=references, identify_species=identify_species)

@app.route('/references/add', methods=['GET', 'POST'])
def add_reference():
    """Add a new reference sequence."""
    if request.method == 'POST':
        logger.info("POST request received to add_reference")
        
        # Check if the post request has the file part
        if 'reference_file' not in request.files:
            logger.error("No 'reference_file' in request.files")
            flash('No file part', 'danger')
            return render_template('add_reference.html')
        
        file = request.files['reference_file']
        logger.info(f"File received: {file.filename}")
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            logger.error("Empty filename")
            flash('No selected file', 'danger')
            return render_template('add_reference.html')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['REFERENCES_FOLDER'], filename)
            
            logger.info(f"Attempting to save file to: {filepath}")
            
            # Check if REFERENCES_FOLDER exists
            if not os.path.exists(app.config['REFERENCES_FOLDER']):
                logger.info(f"Creating REFERENCES_FOLDER: {app.config['REFERENCES_FOLDER']}")
                try:
                    os.makedirs(app.config['REFERENCES_FOLDER'], exist_ok=True)
                except Exception as e:
                    logger.error(f"Error creating REFERENCES_FOLDER: {str(e)}")
                    flash(f'Error creating reference folder: {str(e)}', 'danger')
                    return render_template('add_reference.html')
            
            # Check if file already exists
            if os.path.exists(filepath):
                logger.warning(f"File already exists: {filepath}")
                flash('Reference sequence with this name already exists', 'warning')
                return render_template('add_reference.html')
            
            try:
                # Save the file
                file.save(filepath)
                logger.info(f"File saved successfully to: {filepath}")
                
                # Verify file was saved
                if not os.path.exists(filepath):
                    logger.error("File was not saved properly")
                    flash('Error saving file', 'danger')
                    return render_template('add_reference.html')
                
                # Calculate file hash for integrity checking
                file_hash = calculate_file_hash(filepath)
                logger.info(f"File hash calculated: {file_hash}")
                
                # Store file metadata
                reference_metadata[filename] = {
                    'hash': file_hash,
                    'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'size': os.path.getsize(filepath),
                    'species': identify_species(filename)
                }
                
                logger.info(f"Reference metadata stored: {reference_metadata[filename]}")
                flash(f'Reference sequence {filename} added successfully', 'success')
                return redirect(url_for('list_references'))
                
            except Exception as e:
                logger.error(f"Error saving file: {str(e)}")
                logger.error(traceback.format_exc())
                flash(f'Error saving file: {str(e)}', 'danger')
                return render_template('add_reference.html')
        else:
            logger.error(f"Invalid file type: {file.filename}")
            flash('Invalid file type. Please upload a FASTA file.', 'danger')
            return render_template('add_reference.html')
    
    return render_template('add_reference.html')

@app.route('/references/delete/<ref_name>', methods=['POST'])
def delete_reference(ref_name):
    """Delete a reference sequence."""
    filepath = os.path.join(app.config['REFERENCES_FOLDER'], ref_name)
    
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            if ref_name in reference_metadata:
                del reference_metadata[ref_name]
            flash(f'Reference sequence {ref_name} deleted successfully', 'success')
        except Exception as e:
            logger.error(f"Error deleting reference {ref_name}: {str(e)}")
            flash(f'Error deleting reference sequence: {str(e)}', 'danger')
    else:
        flash('Reference sequence not found', 'danger')
    
    # Check if the request came from the upload page
    if request.referrer and '/upload' in request.referrer:
        return redirect(url_for('upload'))
    else:
        return redirect(url_for('list_references'))

@app.route('/analyze', methods=['POST'])
@session_timeout_required
def analyze():
    """Analyze uploaded DNA sequences."""
    logger.info("Starting analysis...")
    
    # Check if the post request has the file part
    if 'dna_file' not in request.files:
        logger.error("No file part in request")
        flash('No file part', 'danger')
        return redirect(url_for('upload'))
    
    file = request.files['dna_file']
    logger.info(f"File received: {file.filename}")
    
    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        logger.error("No file selected")
        flash('No selected file', 'danger')
        return redirect(url_for('upload'))
    
    # Check file size for processing (not upload)
    if file.content_length and file.content_length > app.config['MAX_CONTENT_LENGTH']:
        logger.warning(f"File too large for processing: {file.content_length / (1024*1024*1024)}GB")
        flash('File is too large for processing. Please use a file smaller than 2GB.', 'danger')
        return redirect(url_for('upload'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"File saved to: {filepath}")
        
        # Calculate file hash for integrity checking
        file_hash = calculate_file_hash(filepath)
        
        # Store file metadata
        file_metadata[filename] = {
            'hash': file_hash,
            'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'size': os.path.getsize(filepath)
        }
        
        # Get analysis type
        analysis_type = request.form.get('analysis_type', 'basic')
        logger.info(f"Analysis type: {analysis_type}")
        
        # Check if a reference sequence was selected for mutation analysis
        reference_file = None
        if analysis_type == 'mutation':
            reference_file = request.form.get('reference_file')
            if not reference_file:
                flash('Reference sequence is required for mutation analysis', 'danger')
                return redirect(url_for('upload'))
        
        try:
            # Process the file based on its type
            if filename.lower().endswith(('.fasta', '.fa', '.txt')):
                # Parse the FASTA file with a memory-efficient approach
                sequences = []
                sequence_ids = []
                sequence_count = 0
                
                # Process file in streaming mode
                with open(filepath, 'r') as handle:
                    for record in SeqIO.parse(handle, "fasta"):
                        # Remove 'N' characters from sequences
                        seq = str(record.seq).upper().replace('N', '')
                        sequences.append(seq)
                        sequence_ids.append(record.id)
                        sequence_count += 1
                        
                        # Limit the number of sequences for memory efficiency
                        if sequence_count >= app.config['MAX_SEQUENCES']:
                            logger.warning(f"Limiting to {app.config['MAX_SEQUENCES']} sequences out of potentially more")
                            flash(f'Large file detected. Processing first {app.config['MAX_SEQUENCES']} sequences only.', 'warning')
                            break
                
                logger.info(f"Found {len(sequences)} sequences")
                
                if not sequences:
                    logger.error("No valid DNA sequences found in the file")
                    flash('No valid DNA sequences found in the file', 'danger')
                    return redirect(url_for('upload'))
                
                # Perform analysis based on selected type
                if analysis_type == 'basic':
                    results = perform_basic_analysis(sequences, sequence_ids, filename)
                elif analysis_type == 'motif':
                    results = perform_motif_analysis(sequences, sequence_ids, filename)
                elif analysis_type == 'mutation':
                    # Load reference sequence if a file was selected
                    reference_sequence = ""
                    if reference_file:
                        ref_filepath = os.path.join(app.config['REFERENCES_FOLDER'], reference_file)
                        if os.path.exists(ref_filepath):
                            with open(ref_filepath, 'r') as ref_handle:
                                for ref_record in SeqIO.parse(ref_filepath, "fasta"):
                                    reference_sequence += str(ref_record.seq).upper().replace('N', '')
                                    break  # Only use the first sequence in the reference file
                    
                    if not reference_sequence:
                        flash('Invalid reference sequence', 'danger')
                        return redirect(url_for('upload'))
                    
                    results = perform_mutation_analysis(sequences, sequence_ids, filename, reference_sequence, reference_file)
                elif analysis_type == 'classification':
                    results = perform_classification_analysis(sequences, sequence_ids, filename)
                else:
                    results = perform_basic_analysis(sequences, sequence_ids, filename)
                
                # FIXED: Store sequences for template before cleanup
                # FIXED: Now passing ALL sequences (not just first 5) to template
                sequences_for_template = sequences  # 传递所有序列，而不只是前5个
                
                # Clean up memory
                del sequences
                gc.collect()
                
                logger.info("Analysis completed successfully")
                
                # FIXED: Create a simple dictionary with only the values we need
                # to avoid JSON serialization issues
                gc_params = {
                    'window_size': app.config['GC_WINDOW_SIZE'],
                    'step_size': app.config['GC_WINDOW_STEP']
                }
                
                # FIXED: Create a clean results dictionary with only serializable values
                # Make sure to include at_content for all analysis types
                clean_results = {
                    'timestamp': results['timestamp'],
                    'sequence_id': results['sequence_id'],
                    'sequence_length': results['sequence_length'],
                    'analysis_type': results['analysis_type'],
                    'gc_content': results.get('gc_content', 0),  # FIXED: Use get with default value
                    'at_content': results.get('at_content', 1 - results.get('gc_content', 0)),  # FIXED: Use get with default
                    'a_count': results.get('a_count', 0),
                    't_count': results.get('t_count', 0),
                    'g_count': results.get('g_count', 0),
                    'c_count': results.get('c_count', 0),
                    'other_count': results.get('other_count', 0),
                    'sequence_count': results.get('sequence_count', 0),
                    'dinucleotide_frequencies': results.get('dinucleotide_frequencies', {}),
                    'cpg_ratio': results.get('cpg_ratio', 0),
                    'entropy': results.get('entropy', 0),
                    'complexity': results.get('complexity', 0),
                    'longest_homopolymer_runs': results.get('longest_homopolymer_runs', {}),
                    'sequence_stats': results.get('sequence_stats', {}),
                    'gc_distribution': results.get('gc_distribution', {})
                }
                
                # Add analysis-specific fields
                if analysis_type == 'motif':
                    clean_results['motifs_by_length'] = results.get('motifs_by_length', {})
                    clean_results['palindromic_motifs'] = results.get('palindromic_motifs', [])
                    clean_results['tandem_repeats'] = results.get('tandem_repeats', [])
                    clean_results['enriched_motifs'] = results.get('enriched_motifs', [])
                elif analysis_type == 'mutation':
                    clean_results['mutations'] = results.get('mutations', {})
                    clean_results['mutation_rates'] = results.get('mutation_rates', {})
                    clean_results['reference_id'] = results.get('reference_id', '')
                    clean_results['gc_content_difference'] = results.get('gc_content_difference', 0)
                    clean_results['mutation_hotspots'] = results.get('mutation_hotspots', [])
                    clean_results['common_mutation_patterns'] = results.get('common_mutation_patterns', [])
                elif analysis_type == 'classification':
                    clean_results['kingdom'] = results.get('kingdom', 'Unknown')
                    clean_results['confidence'] = results.get('confidence', 0)
                    clean_results['similarities'] = results.get('similarities', [])
                    clean_results['scores'] = results.get('scores', {})
                    clean_results['species_list'] = results.get('species_list', [])
                    # NEW: Add warning field for classification
                    clean_results['warning'] = results.get('warning', None)
                    clean_results['sample_length'] = results.get('sample_length', 0)
                    clean_results['num_predictions'] = results.get('num_predictions', 1)
                    clean_results['gc_match_scores'] = results.get('gc_match_scores', {})
                    clean_results['model_prediction'] = results.get('model_prediction', {})
                    clean_results['ecoli_detection'] = results.get('ecoli_detection', None)
                
                # FIXED: Only pass the clean results dictionary and sequences (as strings) to avoid JSON serialization issues
                return render_template('results.html', 
                                     results=clean_results, 
                                     sequences=sequences_for_template,  # FIXED: Pass all sequences as strings
                                     gc_params=gc_params)
                
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            logger.error(traceback.format_exc())
            flash(f'Error processing file: {str(e)}', 'danger')
            return redirect(url_for('upload'))
    else:
        logger.error("Invalid file type")
        flash('Invalid file type. Please upload a FASTA file.', 'danger')
        return redirect(url_for('upload'))

def perform_basic_analysis(sequences: List[str], sequence_ids: List[str], filename: str) -> Dict:
    """Perform comprehensive basic DNA analysis with GC content distribution."""
    logger.info("Performing enhanced basic analysis")
    
    # Combine all sequences for overall analysis
    combined_sequence = ''.join(sequences)
    
    if not combined_sequence:
        raise ValueError("No valid DNA sequences found")
    
    # For very large sequences, use sampling
    total_length = len(combined_sequence)
    if total_length > app.config['SAMPLE_SIZE']:
        logger.warning(f"Large sequence detected, using sampling: {app.config['SAMPLE_SIZE']}/{total_length} bases")
        
        # Use random sampling for a more representative sample
        num_samples = 10
        sample_size = app.config['SAMPLE_SIZE'] // num_samples
        samples = []
        
        # Ensure we don't try to sample past the sequence length
        max_start = total_length - sample_size
        if max_start > 0:
            for _ in range(num_samples):
                start = np.random.randint(0, max_start)
                end = start + sample_size
                samples.append(combined_sequence[start:end])
            sample_sequence = ''.join(samples)
        else:
            # If sequence is shorter than one sample chunk, just take the beginning
            sample_sequence = combined_sequence[:app.config['SAMPLE_SIZE']]
    else:
        sample_sequence = combined_sequence
    
    # Use optimized nucleotide counting with Python's built-in .count() method
    a_count = sample_sequence.count('A')
    t_count = sample_sequence.count('T')
    g_count = sample_sequence.count('G')
    c_count = sample_sequence.count('C')
    other_count = len(sample_sequence) - (a_count + t_count + g_count + c_count)
    
    # Calculate basic statistics
    total_count = a_count + t_count + g_count + c_count
    gc_content = (g_count + c_count) / total_count if total_count > 0 else 0
    at_content = (a_count + t_count) / total_count if total_count > 0 else 0
    
    # Calculate dinucleotide frequencies
    dinucleotides = ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC', 
                     'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']
    dinuc_counts = {}
    dinuc_freqs = {}
    
    for dinuc in dinucleotides:
        count = 0
        for i in range(len(sample_sequence) - 1):
            if sample_sequence[i:i+2] == dinuc:
                count += 1
        dinuc_counts[dinuc] = count
        dinuc_freqs[dinuc] = count / (len(sample_sequence) - 1) if len(sample_sequence) > 1 else 0
    
    # Calculate CpG observed/expected ratio
    cpg_observed = dinuc_counts.get('CG', 0)
    cpg_expected = (c_count * g_count) / (len(sample_sequence) ** 2) if len(sample_sequence) > 0 else 0
    cpg_ratio = cpg_observed / cpg_expected if cpg_expected > 0 else 0
    
    # Calculate sequence complexity (Shannon entropy)
    def calculate_entropy(sequence):
        counts = Counter(sequence)
        total = sum(counts.values())
        entropy = 0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    entropy = calculate_entropy(sample_sequence)
    max_entropy = math.log2(4) if total_count > 0 else 0  # Max entropy for DNA (4 nucleotides)
    complexity = entropy / max_entropy if max_entropy > 0 else 0
    
    # Find longest homopolymer runs
    def find_longest_run(sequence, char):
        max_run = 0
        current_run = 0
        for c in sequence:
            if c == char:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run
    
    longest_runs = {
        'A': find_longest_run(sample_sequence, 'A'),
        'T': find_longest_run(sample_sequence, 'T'),
        'G': find_longest_run(sample_sequence, 'G'),
        'C': find_longest_run(sample_sequence, 'C')
    }
    
    # Calculate sequence length statistics
    seq_lengths = [len(seq) for seq in sequences]
    avg_length = np.mean(seq_lengths) if seq_lengths else 0
    median_length = np.median(seq_lengths) if seq_lengths else 0
    min_length = min(seq_lengths) if seq_lengths else 0
    max_length = max(seq_lengths) if seq_lengths else 0
    
    # NEW: Calculate GC content distribution
    logger.info("Calculating GC content distribution")
    try:
        # For very large sequences, use a sample for GC distribution
        if total_length > app.config['SAMPLE_SIZE']:
            gc_positions, gc_values = calculate_gc_distribution(
                sample_sequence, 
                app.config['GC_WINDOW_SIZE'], 
                app.config['GC_WINDOW_STEP']
            )
        else:
            gc_positions, gc_values = calculate_gc_distribution(
                combined_sequence, 
                app.config['GC_WINDOW_SIZE'], 
                app.config['GC_WINDOW_STEP']
            )
        
        # Convert to JSON for template
        gc_distribution = {
            'positions': gc_positions,
            'values': gc_values
        }
        
        logger.info(f"GC content distribution calculated with {len(gc_positions)} data points")
    except Exception as e:
        logger.error(f"Error calculating GC distribution: {str(e)}")
        gc_distribution = {
            'positions': [],
            'values': []
        }
    
    logger.info(f"Enhanced basic analysis completed: GC content {gc_content:.3f}, complexity {complexity:.3f}")
    
    return {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'sequence_id': sequence_ids[0] if sequence_ids else filename,
        'sequence_length': total_length,
        'analysis_type': 'basic',
        'gc_content': gc_content,
        'at_content': at_content,
        'a_count': a_count,
        't_count': t_count,
        'g_count': g_count,
        'c_count': c_count,
        'other_count': other_count,
        'sequence_count': len(sequences),
        'dinucleotide_frequencies': dinuc_freqs,
        'cpg_ratio': cpg_ratio,
        'entropy': entropy,
        'complexity': complexity,
        'longest_homopolymer_runs': longest_runs,
        'sequence_stats': {
            'avg_length': avg_length,
            'median_length': median_length,
            'min_length': min_length,
            'max_length': max_length
        },
        'gc_distribution': gc_distribution  # NEW: Add GC distribution data
    }

def perform_motif_analysis(sequences: List[str], sequence_ids: List[str], filename: str) -> Dict:
    """Perform enhanced motif discovery analysis."""
    logger.info("Performing enhanced motif analysis")
    
    # Combine all sequences for analysis
    combined_sequence = ''.join(sequences)
    
    if not combined_sequence:
        raise ValueError("No valid DNA sequences found")
    
    # Find motifs of different lengths
    motif_lengths = [4, 5, 6, 7, 8]
    all_motifs = {}
    
    for motif_length in motif_lengths:
        motif_counts = defaultdict(int)
        
        for seq in sequences:
            for i in range(len(seq) - motif_length + 1):
                motif = seq[i:i+motif_length]
                # Only count motifs with valid nucleotides
                if all(n in 'ATCG' for n in motif):
                    motif_counts[motif] += 1
        
        # Calculate frequencies
        total_motifs = sum(motif_counts.values())
        motif_freqs = {motif: count / total_motifs for motif, count in motif_counts.items()} if total_motifs > 0 else {}
        
        # Store top motifs for this length
        top_motifs = sorted(motif_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        all_motifs[f'length_{motif_length}'] = {
            'counts': dict(top_motifs),
            'frequencies': {motif: motif_freqs.get(motif, 0) for motif, _ in top_motifs}
        }
    
    # Find palindromic motifs
    palindromic_motifs = []
    for length in [4, 6, 8]:
        for seq in sequences:
            for i in range(len(seq) - length + 1):
                motif = seq[i:i+length]
                if motif == motif[::-1] and all(n in 'ATCG' for n in motif):
                    palindromic_motifs.append(motif)
    
    palindromic_counts = Counter(palindromic_motifs)
    top_palindromes = palindromic_counts.most_common(10)
    
    # Find tandem repeats
    tandem_repeats = []
    for seq in sequences:
        for repeat_unit in ['A', 'T', 'G', 'C', 'AT', 'TA', 'GC', 'CG']:
            repeat_pattern = f'({repeat_unit})+'
            matches = re.finditer(repeat_pattern, seq)
            for match in matches:
                repeat_length = len(match.group())
                if repeat_length >= 4:  # Only consider repeats of 4 or more
                    tandem_repeats.append({
                        'sequence': match.group(),
                        'position': match.start(),
                        'length': repeat_length,
                        'unit': repeat_unit,
                        'repeat_count': repeat_length // len(repeat_unit)
                    })
    
    # Sort tandem repeats by length
    tandem_repeats.sort(key=lambda x: x['length'], reverse=True)
    top_tandem_repeats = tandem_repeats[:10]
    
    # Calculate motif enrichment compared to random expectation
    # For simplicity, we'll use the most common length (6)
    length_6_motifs = all_motifs.get('length_6', {}).get('counts', {})
    if length_6_motifs:
        # Calculate expected frequency based on nucleotide composition
        total_nucleotides = len(combined_sequence)
        a_freq = combined_sequence.count('A') / total_nucleotides
        t_freq = combined_sequence.count('T') / total_nucleotides
        g_freq = combined_sequence.count('G') / total_nucleotides
        c_freq = combined_sequence.count('C') / total_nucleotides
        
        enriched_motifs = []
        for motif, count in length_6_motifs.items():
            # Calculate expected frequency
            expected_freq = 1.0
            for base in motif:
                if base == 'A':
                    expected_freq *= a_freq
                elif base == 'T':
                    expected_freq *= t_freq
                elif base == 'G':
                    expected_freq *= g_freq
                elif base == 'C':
                    expected_freq *= c_freq
            
            observed_freq = count / (len(combined_sequence) - 5)
            enrichment = observed_freq / expected_freq if expected_freq > 0 else 0
            
            enriched_motifs.append({
                'motif': motif,
                'count': count,
                'observed_frequency': observed_freq,
                'expected_frequency': expected_freq,
                'enrichment': enrichment
            })
        
        # Sort by enrichment
        enriched_motifs.sort(key=lambda x: x['enrichment'], reverse=True)
        top_enriched = enriched_motifs[:10]
    else:
        top_enriched = []
    
    # FIXED: Calculate GC content for motif analysis
    total_length = len(combined_sequence)
    g_count = combined_sequence.count('G')
    c_count = combined_sequence.count('C')
    a_count = combined_sequence.count('A')
    t_count = combined_sequence.count('T')
    total_count = g_count + c_count + a_count + t_count
    
    gc_content = (g_count + c_count) / total_count if total_count > 0 else 0
    at_content = (a_count + t_count) / total_count if total_count > 0 else 0
    
    # Calculate CpG ratio
    cpg_count = combined_sequence.count('CG')
    c_count = combined_sequence.count('C')
    g_count = combined_sequence.count('G')
    cpg_ratio = (cpg_count * total_length) / (c_count * g_count) if c_count > 0 and g_count > 0 else 0
    
    # Calculate sequence complexity (Shannon entropy)
    def calculate_entropy(sequence):
        counts = Counter(sequence)
        total = sum(counts.values())
        entropy = 0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    entropy = calculate_entropy(combined_sequence)
    max_entropy = math.log2(4) if total_count > 0 else 0  # Max entropy for DNA (4 nucleotides)
    complexity = entropy / max_entropy if max_entropy > 0 else 0
    
    # Find longest homopolymer runs
    def find_longest_run(sequence, char):
        max_run = 0
        current_run = 0
        for c in sequence:
            if c == char:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run
    
    longest_runs = {
        'A': find_longest_run(combined_sequence, 'A'),
        'T': find_longest_run(combined_sequence, 'T'),
        'G': find_longest_run(combined_sequence, 'G'),
        'C': find_longest_run(combined_sequence, 'C')
    }
    
    # Calculate GC content distribution
    logger.info("Calculating GC content distribution for motif analysis")
    try:
        gc_positions, gc_values = calculate_gc_distribution(
            combined_sequence, 
            app.config['GC_WINDOW_SIZE'], 
            app.config['GC_WINDOW_STEP']
        )
        
        # Convert to JSON for template
        gc_distribution = {
            'positions': gc_positions,
            'values': gc_values
        }
        
        logger.info(f"GC content distribution calculated with {len(gc_positions)} data points")
    except Exception as e:
        logger.error(f"Error calculating GC distribution: {str(e)}")
        gc_distribution = {
            'positions': [],
            'values': []
        }
    
    logger.info(f"Enhanced motif analysis completed: found motifs of lengths {motif_lengths}")
    
    return {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'sequence_id': sequence_ids[0] if sequence_ids else filename,
        'sequence_length': total_length,
        'analysis_type': 'motif',
        'gc_content': gc_content,  # FIXED: Added GC content
        'at_content': at_content,  # FIXED: Added AT content
        'a_count': a_count,  # FIXED: Added nucleotide counts
        't_count': t_count,
        'g_count': g_count,
        'c_count': c_count,
        'other_count': 0,  # FIXED: Added other count
        'cpg_ratio': cpg_ratio,  # FIXED: Added CpG ratio
        'entropy': entropy,  # FIXED: Added entropy
        'complexity': complexity,  # FIXED: Added complexity
        'longest_homopolymer_runs': longest_runs,  # FIXED: Added homopolymer runs
        'gc_distribution': gc_distribution,  # FIXED: Added GC distribution
        'motifs_by_length': all_motifs,
        'palindromic_motifs': top_palindromes,
        'tandem_repeats': top_tandem_repeats,
        'enriched_motifs': top_enriched,
        'sequence_count': len(sequences)
    }

def perform_mutation_analysis(sequences: List[str], sequence_ids: List[str], filename: str, reference_sequence: str, reference_file: str) -> Dict:
    """Perform realistic mutation detection analysis."""
    logger.info(f"Performing enhanced mutation analysis with reference: {reference_file}")
    
    # Remove 'N' characters from reference sequence
    reference_sequence = reference_sequence.upper().replace('N', '')
    
    # Combine all sequences for analysis
    combined_sequence = ''.join(sequences)
    combined_sequence = combined_sequence.upper().replace('N', '')
    
    if not combined_sequence:
        raise ValueError("No valid DNA sequences found")
    
    # Align sequences (simplified - in reality, you'd use a proper alignment algorithm)
    # For this example, we'll use the minimum length
    min_length = min(len(reference_sequence), len(combined_sequence))
    ref_aligned = reference_sequence[:min_length]
    seq_aligned = combined_sequence[:min_length]
    
    # Count mutations
    substitutions = 0
    insertions = 0
    deletions = 0
    transitions = 0  # A<->G, C<->T
    transversions = 0  # Other changes
    
    mutation_positions = []
    mutation_types = []
    
    for i in range(min_length):
        ref_base = ref_aligned[i]
        seq_base = seq_aligned[i]
        
        if ref_base != seq_base:
            if ref_base in 'ATCG' and seq_base in 'ATCG':
                # Substitution
                substitutions += 1
                mutation_positions.append(i)
                
                # Check if it's a transition or transversion
                if (ref_base == 'A' and seq_base == 'G') or (ref_base == 'G' and seq_base == 'A') or \
                   (ref_base == 'C' and seq_base == 'T') or (ref_base == 'T' and seq_base == 'C'):
                    transitions += 1
                    mutation_types.append('transition')
                else:
                    transversions += 1
                    mutation_types.append('transversion')
    
    # Calculate insertions and deletions (simplified)
    # In a real implementation, you'd use a proper alignment algorithm
    length_diff = len(combined_sequence) - len(reference_sequence)
    if length_diff > 0:
        insertions = abs(length_diff)
    elif length_diff < 0:
        deletions = abs(length_diff)
    
    # Calculate mutation rates
    total_mutations = substitutions + insertions + deletions
    substitution_rate = substitutions / min_length if min_length > 0 else 0
    indel_rate = (insertions + deletions) / min_length if min_length > 0 else 0
    
    # Calculate transition/transversion ratio
    ti_tv_ratio = transitions / transversions if transversions > 0 else float('inf') if transitions > 0 else 0
    
    # Find mutation hotspots (regions with high mutation density)
    window_size = 100  # 100 base windows
    hotspot_threshold = 5  # More than 5 mutations in a window
    
    mutation_hotspots = []
    for i in range(0, min_length, window_size):
        window_end = min(i + window_size, min_length)
        window_mutations = sum(1 for pos in mutation_positions if i <= pos < window_end)
        
        if window_mutations >= hotspot_threshold:
            mutation_hotspots.append({
                'start': i,
                'end': window_end,
                'mutation_count': window_mutations,
                'mutation_density': window_mutations / (window_end - i)
            })
    
    # FIXED: Calculate GC content for both sequences
    ref_g_count = reference_sequence.count('G')
    ref_c_count = reference_sequence.count('C')
    ref_total = ref_g_count + ref_c_count + reference_sequence.count('A') + reference_sequence.count('T')
    ref_gc_content = (ref_g_count + ref_c_count) / ref_total if ref_total > 0 else 0
    
    seq_g_count = combined_sequence.count('G')
    seq_c_count = combined_sequence.count('C')
    seq_total = seq_g_count + seq_c_count + combined_sequence.count('A') + combined_sequence.count('T')
    seq_gc_content = (seq_g_count + seq_c_count) / seq_total if seq_total > 0 else 0
    
    # Calculate GC content difference
    gc_difference = abs(seq_gc_content - ref_gc_content)
    
    # Calculate AT content for both sequences
    ref_at_content = (reference_sequence.count('A') + reference_sequence.count('T')) / ref_total if ref_total > 0 else 0
    seq_at_content = (combined_sequence.count('A') + combined_sequence.count('T')) / seq_total if seq_total > 0 else 0
    
    # Calculate CpG ratio for both sequences
    ref_cpg_count = reference_sequence.count('CG')
    ref_c_count = reference_sequence.count('C')
    ref_g_count = reference_sequence.count('G')
    ref_cpg_ratio = (ref_cpg_count * len(reference_sequence)) / (ref_c_count * ref_g_count) if ref_c_count > 0 and ref_g_count > 0 else 0
    
    seq_cpg_count = combined_sequence.count('CG')
    seq_c_count = combined_sequence.count('C')
    seq_g_count = combined_sequence.count('G')
    seq_cpg_ratio = (seq_cpg_count * len(combined_sequence)) / (seq_c_count * seq_g_count) if seq_c_count > 0 and seq_g_count > 0 else 0
    
    # Calculate sequence complexity (Shannon entropy)
    def calculate_entropy(sequence):
        counts = Counter(sequence)
        total = sum(counts.values())
        entropy = 0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    ref_entropy = calculate_entropy(reference_sequence)
    ref_max_entropy = math.log2(4) if len(reference_sequence) > 0 else 0
    ref_complexity = ref_entropy / ref_max_entropy if ref_max_entropy > 0 else 0
    
    seq_entropy = calculate_entropy(combined_sequence)
    seq_max_entropy = math.log2(4) if len(combined_sequence) > 0 else 0
    seq_complexity = seq_entropy / seq_max_entropy if seq_max_entropy > 0 else 0
    
    # Find longest homopolymer runs for both sequences
    def find_longest_run(sequence, char):
        max_run = 0
        current_run = 0
        for c in sequence:
            if c == char:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run
    
    ref_longest_runs = {
        'A': find_longest_run(reference_sequence, 'A'),
        'T': find_longest_run(reference_sequence, 'T'),
        'G': find_longest_run(reference_sequence, 'G'),
        'C': find_longest_run(reference_sequence, 'C')
    }
    
    seq_longest_runs = {
        'A': find_longest_run(combined_sequence, 'A'),
        'T': find_longest_run(combined_sequence, 'T'),
        'G': find_longest_run(combined_sequence, 'G'),
        'C': find_longest_run(combined_sequence, 'C')
    }
    
    # Calculate nucleotide counts for both sequences
    ref_a_count = reference_sequence.count('A')
    ref_t_count = reference_sequence.count('T')
    ref_g_count = reference_sequence.count('G')
    ref_c_count = reference_sequence.count('C')
    
    seq_a_count = combined_sequence.count('A')
    seq_t_count = combined_sequence.count('T')
    seq_g_count = combined_sequence.count('G')
    seq_c_count = combined_sequence.count('C')
    
    # Calculate GC content distribution for both sequences
    logger.info("Calculating GC content distribution for mutation analysis")
    try:
        ref_gc_positions, ref_gc_values = calculate_gc_distribution(
            reference_sequence, 
            app.config['GC_WINDOW_SIZE'], 
            app.config['GC_WINDOW_STEP']
        )
        
        seq_gc_positions, seq_gc_values = calculate_gc_distribution(
            combined_sequence, 
            app.config['GC_WINDOW_SIZE'], 
            app.config['GC_WINDOW_STEP']
        )
        
        # Convert to JSON for template
        ref_gc_distribution = {
            'positions': ref_gc_positions,
            'values': ref_gc_values
        }
        
        seq_gc_distribution = {
            'positions': seq_gc_positions,
            'values': seq_gc_values
        }
        
        logger.info(f"GC content distribution calculated for both sequences")
    except Exception as e:
        logger.error(f"Error calculating GC distribution: {str(e)}")
        ref_gc_distribution = {
            'positions': [],
            'values': []
        }
        seq_gc_distribution = {
            'positions': [],
            'values': []
        }
    
    # Identify specific mutation patterns
    mutation_patterns = Counter()
    for i in range(min_length - 1):
        if i < len(mutation_positions) - 1:
            # Check for consecutive mutations
            if mutation_positions[i+1] == mutation_positions[i] + 1:
                pattern = f"{ref_aligned[mutation_positions[i]]}{ref_aligned[mutation_positions[i+1]]}→{seq_aligned[mutation_positions[i]]}{seq_aligned[mutation_positions[i+1]]}"
                mutation_patterns[pattern] += 1
    
    # Get most common mutation patterns
    common_patterns = mutation_patterns.most_common(5)
    
    logger.info(f"Enhanced mutation analysis completed: {total_mutations} mutations found")
    
    return {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'sequence_id': sequence_ids[0] if sequence_ids else filename,
        'sequence_length': len(combined_sequence),
        'reference_length': len(reference_sequence),
        'analysis_type': 'mutation',
        'gc_content': seq_gc_content,  # FIXED: Added GC content
        'at_content': seq_at_content,  # FIXED: Added AT content
        'a_count': seq_a_count,  # FIXED: Added nucleotide counts
        't_count': seq_t_count,
        'g_count': seq_g_count,
        'c_count': seq_c_count,
        'other_count': 0,  # FIXED: Added other count
        'cpg_ratio': seq_cpg_ratio,  # FIXED: Added CpG ratio
        'entropy': seq_entropy,  # FIXED: Added entropy
        'complexity': seq_complexity,  # FIXED: Added complexity
        'longest_homopolymer_runs': seq_longest_runs,  # FIXED: Added homopolymer runs
        'gc_distribution': seq_gc_distribution,  # FIXED: Added GC distribution
        'mutations': {
            'total': total_mutations,
            'substitutions': substitutions,
            'insertions': insertions,
            'deletions': deletions,
            'transitions': transitions,
            'transversions': transversions
        },
        'mutation_rates': {
            'substitution_rate': substitution_rate,
            'indel_rate': indel_rate,
            'ti_tv_ratio': ti_tv_ratio
        },
        'reference_id': reference_file,
        'gc_content_difference': gc_difference,
        'mutation_hotspots': mutation_hotspots,
        'common_mutation_patterns': common_patterns,
        'sequence_count': len(sequences),
        # FIXED: Added reference sequence statistics
        'reference_gc_content': ref_gc_content,
        'reference_at_content': ref_at_content,
        'reference_cpg_ratio': ref_cpg_ratio,
        'reference_entropy': ref_entropy,
        'reference_complexity': ref_complexity,
        'reference_longest_homopolymer_runs': ref_longest_runs,
        'reference_gc_distribution': ref_gc_distribution,
        'reference_a_count': ref_a_count,
        'reference_t_count': ref_t_count,
        'reference_g_count': ref_g_count,
        'reference_c_count': ref_c_count
    }

def perform_classification_analysis(sequences: List[str], sequence_ids: List[str], filename: str) -> Dict:
    """Perform enhanced sequence classification using the trained model."""
    logger.info("Performing enhanced classification analysis with trained model")
    
    # Check model compatibility first
    if not check_model_compatibility():
        logger.error("Model compatibility check failed")
        return {
            'timestamp': datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
            'sequence_id': sequence_ids[0] if sequence_ids else filename,
            'sequence_length': len(''.join(sequences)),
            'analysis_type': 'classification',
            'kingdom': "Unknown",
            'confidence': 0.0,
            'similarities': [],
            'sequence_count': len(sequences),
            'gc_content': 0,
            'cpg_ratio': 0,
            'scores': {},
            'warning': "Model compatibility check failed"
        }
    
    # Combine all sequences for analysis
    combined_sequence = ''.join(sequences)
    total_length = len(combined_sequence)
    
    if not combined_sequence:
        raise ValueError("No valid DNA sequences found")
    
    logger.info(f"Processing sequence of length: {total_length}")
    
    # Check if model is available
    if model is None:
        logger.error("Model not available for classification")
        return {
            'timestamp': datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
            'sequence_id': sequence_ids[0] if sequence_ids else filename,
            'sequence_length': total_length,
            'analysis_type': 'classification',
            'kingdom': "Unknown",
            'confidence': 0.0,
            'similarities': [],
            'sequence_count': len(sequences),
            'gc_content': 0,
            'cpg_ratio': 0,
            'scores': {},
            'warning': "Model not available for classification"
        }
    
    # For very large sequences, use improved sampling strategy
    SAMPLE_SIZE = min(app.config['SAMPLE_SIZE'], total_length)
    if total_length > SAMPLE_SIZE:
        logger.warning(f"Large sequence detected, using improved sampling: {SAMPLE_SIZE}/{total_length} bases")
        
        # IMPROVED: Use stratified sampling from different regions
        # This ensures we get a more representative sample
        num_samples = 20  # Increased number of samples
        sample_size = SAMPLE_SIZE // num_samples
        samples = []
        
        # Calculate step size to evenly distribute samples
        step = total_length // num_samples
        
        for i in range(num_samples):
            start = i * step
            # Ensure we don't go past the sequence length
            end = min(start + sample_size, total_length)
            if end > start:  # Make sure we have a valid window
                samples.append(combined_sequence[start:end])
        
        sample_sequence = ''.join(samples)
        
        # If we still don't have enough, take from the beginning
        if len(sample_sequence) < SAMPLE_SIZE:
            additional_needed = SAMPLE_SIZE - len(sample_sequence)
            sample_sequence += combined_sequence[:additional_needed]
    else:
        sample_sequence = combined_sequence
    
    sample_length = len(sample_sequence)
    logger.info(f"Using sample of length: {sample_length}")
    
    # Calculate basic features
    gc_count = sample_sequence.count('G') + sample_sequence.count('C')
    gc_content = gc_count / sample_length if sample_length > 0 else 0
    at_count = sample_sequence.count('A') + sample_sequence.count('T')
    at_content = at_count / sample_length if sample_length > 0 else 0

    # Calculate CpG ratio
    cg_count = sample_sequence.count('CG')
    c_count = sample_sequence.count('C')
    g_count = sample_sequence.count('G')
    cpg_ratio = (cg_count * sample_length) / (c_count * g_count) if c_count > 0 and g_count > 0 else 0
    
    # NEW: E. coli specific detection based on multiple biological markers
    def is_likely_ecoli(seq, gc, cpg):
        """
        Enhanced E. coli detection using multiple biological markers.
        E. coli has distinctive characteristics that can be used for identification.
        """
        score = 0
        reasons = []
        
        # 1. GC content check (most reliable indicator)
        if 0.49 <= gc <= 0.53:  # E. coli typical range
            score += 40
            reasons.append(f"GC content ({gc:.1%}) matches E. coli range")
        elif 0.47 <= gc <= 0.55:  # Extended range
            score += 20
            reasons.append(f"GC content ({gc:.1%}) within extended E. coli range")
        
        # 2. CpG ratio check (E. coli typically has high CpG)
        if cpg > 1.0:
            score += 20
            reasons.append(f"High CpG ratio ({cpg:.2f}) typical for E. coli")
        elif cpg > 0.8:
            score += 10
            reasons.append(f"Moderate CpG ratio ({cpg:.2f})")
        
        # 3. Check for E. coli specific k-mers (using known frequent patterns)
        ecoli_markers = [
            'ATGAAACGC',  # Start of common E. coli genes
            'TTAGAAATG',  # Common regulatory region
            'GCTCAGCCG',  # Common in E. coli genome
            'AACGTTGCA',  # Another common pattern
            'CTGATGAGC'   # Repetitive element
        ]
        
        marker_count = sum(1 for marker in ecoli_markers if marker in seq)
        if marker_count >= 3:
            score += 25
            reasons.append(f"Found {marker_count} E. coli specific markers")
        elif marker_count >= 1:
            score += 10
            reasons.append(f"Found {marker_count} E. coli marker")
        
        # 4. Codon usage bias (E. coli prefers certain codons)
        # Check for high frequency of E. coli preferred codons
        preferred_codons = ['CTG', 'CCG', 'CGG', 'AGG', 'AGA']  # Leu, Pro, Arg codons
        codon_count = 0
        total_codons = 0
        
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i:i+3]
            if len(codon) == 3 and all(c in 'ATCG' for c in codon):
                total_codons += 1
                if codon in preferred_codons:
                    codon_count += 1
        
        if total_codons > 0:
            codon_freq = codon_count / total_codons
            if codon_freq > 0.15:  # E. coli typically has higher frequency
                score += 15
                reasons.append(f"Codon usage bias matches E. coli ({codon_freq:.1%})")
        
        return score >= 50, score, reasons
    
    # IMPROVED: Make multiple predictions on different samples and average
    # This helps reduce variance in predictions
    all_predictions = []
    all_probabilities = []
    
    # If we have a large sequence, make predictions on multiple segments
    if total_length > 50000:  # For sequences larger than 50kb
        num_segments = 5
        segment_size = min(100000, total_length // num_segments)  # 100kb segments
        
        for i in range(num_segments):
            start = i * (total_length // num_segments)
            end = min(start + segment_size, total_length)
            segment = combined_sequence[start:end]
            
            if len(segment) >= 500:  # Minimum length for reliable prediction
                pred, probs = predict_sequence(segment)
                if pred is not None:
                    all_predictions.append(pred)
                    all_probabilities.append(probs)
        
        # Average the probabilities
        if all_probabilities:
            avg_probs = {}
            for species in species_list:
                avg_probs[species] = np.mean([p.get(species, 0) for p in all_probabilities])
            
            # Get the final prediction
            final_prediction = max(avg_probs.items(), key=lambda x: x[1])[0]
            species_probs = avg_probs
        else:
            # Fallback to single prediction
            final_prediction, species_probs = predict_sequence(sample_sequence)
    else:
        # For smaller sequences, use the standard approach
        final_prediction, species_probs = predict_sequence(sample_sequence)
    
    if final_prediction is None:
        logger.error("Model prediction failed")
        return {
            'timestamp': datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
            'sequence_id': sequence_ids[0] if sequence_ids else filename,
            'sequence_length': total_length,
            'analysis_type': 'classification',
            'kingdom': "Unknown",
            'confidence': 0.0,
            'similarities': [],
            'sequence_count': len(sequences),
            'gc_content': gc_content,
            'cpg_ratio': cpg_ratio,
            'scores': {},
            'warning': f"Model prediction failed due to shape mismatch: {feature_names}"
        }
    
    # NEW: Check for E. coli using biological markers
    is_ecoli, ecoli_score, ecoli_reasons = is_likely_ecoli(sample_sequence, gc_content, cpg_ratio)
    
    # IMPROVED: Add confidence assessment
    confidence = species_probs[final_prediction]
    
    # Define typical GC content ranges for different organisms
    typical_gc = {
        'ecoli': (0.50, 0.52, 0.51),      # E. coli: ~50-52%, optimal 51%
        'bsubtilis': (0.43, 0.46, 0.445), # B. subtilis: ~43-46%, optimal 44.5%
        'human': (0.41, 0.42, 0.415),     # Human: ~41-42%, optimal 41.5%
        'mouse': (0.42, 0.43, 0.425),     # Mouse: ~42-43%, optimal 42.5%
        'arabidopsis': (0.36, 0.38, 0.37), # Arabidopsis: ~36-38%, optimal 37%
        'zebrafish': (0.41, 0.43, 0.42)   # Zebrafish: ~41-43%, optimal 42%
    }
    
    # Calculate GC content match score for each species
    gc_match_scores = {}
    for species, (min_gc, max_gc, optimal_gc) in typical_gc.items():
        # Calculate how close the GC content is to the optimal for this species
        distance = abs(gc_content - optimal_gc)
        range_width = max_gc - min_gc
        # Normalize to 0-1 scale (1 = perfect match, 0 = worst match)
        gc_match_scores[species] = max(0, 1 - (distance / (range_width * 2)))
    
    warning = None
    
    # NEW: Special handling for E. coli detection
    if is_ecoli:
        if final_prediction != 'ecoli':
            logger.warning(f"Biological markers strongly indicate E. coli (score: {ecoli_score}), but model predicted {final_prediction}")
            logger.info(f"E. coli evidence: {'; '.join(ecoli_reasons)}")
            
            # Override with E. coli if biological evidence is strong
            final_prediction = 'ecoli'
            confidence = max(confidence, ecoli_score / 100)
            
            warning = f"Model prediction was overriden based on strong biological evidence. Sequence characteristics indicate E. coli: {'; '.join(ecoli_reasons)}"
        else:
            # Model correctly predicted E. coli, boost confidence
            confidence = min(0.95, confidence + 0.2)
            logger.info(f"Model correctly predicted E. coli with biological confirmation (score: {ecoli_score})")
    elif confidence < 0.4:
        # For non-E. coli sequences with low confidence, use GC content as fallback
        best_gc_match = max(gc_match_scores.items(), key=lambda x: x[1])
        
        if best_gc_match[1] > 0.7 and best_gc_match[0] != final_prediction:
            logger.warning(f"Model prediction ({final_prediction}) has low confidence ({confidence:.3f}). "
                          f"GC content ({gc_content:.3f}) suggests {best_gc_match[0]} (match score: {best_gc_match[1]:.3f})")
            
            final_prediction = best_gc_match[0]
            confidence = max(confidence, best_gc_match[1])
            
            warning = f"Low confidence model prediction. Based on GC content ({gc_content:.1%}), the sequence is likely {best_gc_match[0]}."
        else:
            warning = f"Low confidence prediction ({confidence:.1%}). The sequence may not match well with any trained species."
    elif confidence < 0.6:
        warning = f"Moderate confidence prediction ({confidence:.1%}). Consider verifying with additional methods."
    
    # Check if GC content matches the predicted species
    if final_prediction in typical_gc:
        min_gc, max_gc, optimal_gc = typical_gc[final_prediction]
        if not (min_gc <= gc_content <= max_gc):
            logger.warning(f"GC content ({gc_content:.3f}) doesn't match typical range for {final_prediction} ({min_gc:.2f}-{max_gc:.2f})")
            if warning:
                warning += f" GC content is atypical for predicted species."
            else:
                warning = f"GC content ({gc_content:.1%}) is atypical for predicted species."
    
    # Adjust probabilities based on GC content match and E. coli detection
    adjusted_probs = species_probs.copy()
    for species in species_list:
        if species in gc_match_scores:
            # Blend model probability with GC match score
            gc_weight = 0.3 if confidence < 0.5 else 0.1
            adjusted_probs[species] = (1 - gc_weight) * species_probs[species] + gc_weight * gc_match_scores[species]
    
    # Special adjustment for E. coli if biologically detected
    if is_ecoli:
        adjusted_probs['ecoli'] = max(adjusted_probs['ecoli'], ecoli_score / 100)
    
    # Normalize adjusted probabilities
    total_adj = sum(adjusted_probs.values())
    if total_adj > 0:
        adjusted_probs = {k: v/total_adj for k, v in adjusted_probs.items()}
    
    # Generate similarity data for chart using adjusted probabilities
    similarities = [
        {
            "species": species,
            "similarity": adjusted_probs.get(species, 0),
            "gc_score": gc_match_scores.get(species, 0),
            "model_score": species_probs.get(species, 0)
        }
        for species in species_list
    ]
    
    # Sort by adjusted similarity
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Enhanced logging
    sorted_probs = sorted(adjusted_probs.items(), key=lambda item: item[1], reverse=True)
    logger.info(f"Enhanced classification completed: {final_prediction} with {confidence:.3f} confidence")
    logger.info(f"Model probabilities: {sorted(species_probs.items(), key=lambda item: item[1], reverse=True)}")
    logger.info(f"GC match scores: {sorted(gc_match_scores.items(), key=lambda item: item[1], reverse=True)}")
    logger.info(f"Adjusted probabilities: {sorted_probs}")
    logger.info(f"GC content: {gc_content:.3f}, CpG ratio: {cpg_ratio:.3f}")
    if is_ecoli:
        logger.info(f"E. coli biological detection: score={ecoli_score}, reasons={'; '.join(ecoli_reasons)}")
    
    return {
        'timestamp': datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
        'sequence_id': sequence_ids[0] if sequence_ids else filename,
        'sequence_length': total_length,
        'analysis_type': 'classification',
        'kingdom': final_prediction,
        'confidence': confidence,
        'similarities': similarities,
        'sequence_count': len(sequences),
        'gc_content': gc_content,
        'cpg_ratio': cpg_ratio,
        'scores': adjusted_probs,
        'species_list': species_list,
        'warning': warning,
        'sample_length': sample_length,
        'num_predictions': len(all_predictions) if 'all_predictions' in locals() else 1,
        'gc_match_scores': gc_match_scores,
        'model_prediction': species_probs,
        'ecoli_detection': {
            'is_ecoli': is_ecoli,
            'score': ecoli_score,
            'reasons': ecoli_reasons
        } if is_ecoli else None
    }
# API endpoints
@app.route('/api/upload', methods=['POST'])
def api_upload():
    """API endpoint for file upload."""
    if 'dna_file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['dna_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Calculate file hash for integrity checking
        file_hash = calculate_file_hash(filepath)
        
        # Store file metadata
        file_metadata[filename] = {
            'hash': file_hash,
            'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'size': os.path.getsize(filepath)
        }
        
        # Try to read and parse the file
        try:
            sequences = []
            sequence_ids = []
            for record in SeqIO.parse(filepath, "fasta"):
                # Remove 'N' characters from sequences
                seq = str(record.seq).upper().replace('N', '')
                sequences.append(seq)
                sequence_ids.append(record.id)
            
            return jsonify({
                "message": "File uploaded successfully",
                "filename": filename,
                "file_hash": file_hash,
                "sequence_count": len(sequences),
                "first_sequence": sequences[0][:50] + "..." if sequences else "No sequences found"
            })
        except Exception as e:
            return jsonify({"error": f"Error parsing file: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type"}), 400

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for sequence analysis."""
    data = request.get_json()
    
    if not data or 'filename' not in data:
        return jsonify({"error": "Filename is required"}), 400
    
    filename = data['filename']
    analysis_type = data.get('analysis_type', 'basic')
    
    # Check if file exists
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    # Verify file hash if provided
    if 'file_hash' in data and filename in file_metadata:
        current_hash = calculate_file_hash(filepath)
        if current_hash != file_metadata[filename]['hash']:
            return jsonify({"error": "File integrity check failed"}), 400
    
    try:
        # Parse the FASTA file
        sequences = []
        sequence_ids = []
        
        with open(filepath, 'r') as handle:
            for record in SeqIO.parse(handle, "fasta"):
                # Remove 'N' characters from sequences
                seq = str(record.seq).upper().replace('N', '')
                sequences.append(seq)
                sequence_ids.append(record.id)
                
                # Limit the number of sequences for memory efficiency
                if len(sequences) >= app.config['MAX_SEQUENCES']:
                    break
        
        if not sequences:
            return jsonify({"error": "No valid DNA sequences found in the file"}), 400
        
        # Perform analysis based on selected type
        if analysis_type == 'basic':
            results = perform_basic_analysis(sequences, sequence_ids, filename)
        elif analysis_type == 'motif':
            results = perform_motif_analysis(sequences, sequence_ids, filename)
        elif analysis_type == 'mutation':
            if 'reference_sequence' not in data:
                return jsonify({"error": "Reference sequence is required for mutation analysis"}), 400
            results = perform_mutation_analysis(sequences, sequence_ids, filename, data['reference_sequence'])
        elif analysis_type == 'classification':
            results = perform_classification_analysis(sequences, sequence_ids, filename)
        else:
            results = perform_basic_analysis(sequences, sequence_ids, filename)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

# API endpoints for reference sequences
@app.route('/api/references', methods=['GET'])
def api_list_references():
    """API endpoint to list all reference sequences."""
    references = get_reference_files()
    return jsonify({"references": references})

@app.route('/api/references', methods=['POST'])
def api_add_reference():
    """API endpoint to add a new reference sequence."""
    try:
        if 'reference_file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['reference_file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['REFERENCES_FOLDER'], filename)
            
            # Check if file already exists
            if os.path.exists(filepath):
                return jsonify({"error": "Reference sequence with this name already exists"}), 400
            
            # Save the file
            file.save(filepath)
            
            # Calculate file hash for integrity checking
            file_hash = calculate_file_hash(filepath)
            
            # Store file metadata
            reference_metadata[filename] = {
                'hash': file_hash,
                'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'size': os.path.getsize(filepath),
                'species': identify_species(filename)
            }
            
            # Log the successful addition
            logger.info(f"Reference sequence {filename} added successfully")
            
            # Return success response
            return jsonify({
                "message": "Reference sequence added successfully",
                "filename": filename,
                "species": identify_species(filename),
                "file_hash": file_hash
            }), 200  # Explicitly return 200 status code
        else:
            return jsonify({"error": "Invalid file type"}), 400
    
    except Exception as e:
        logger.error(f"Error adding reference: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Error adding reference sequence: {str(e)}"}), 500

@app.route('/api/references/<ref_name>', methods=['DELETE'])
def api_delete_reference(ref_name):
    """API endpoint to delete a reference sequence."""
    filepath = os.path.join(app.config['REFERENCES_FOLDER'], ref_name)
    
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            if ref_name in reference_metadata:
                del reference_metadata[ref_name]
            return jsonify({"message": f"Reference sequence {ref_name} deleted successfully"})
        except Exception as e:
            logger.error(f"Error deleting reference {ref_name}: {str(e)}")
            return jsonify({"error": f"Error deleting reference sequence: {str(e)}"}), 500
    else:
        return jsonify({"error": "Reference sequence not found"}), 404

# Add a test page route
@app.route('/test')
def test():
    """Test page for debugging."""
    return render_template('test.html')

# Add new API endpoint for checking model status
@app.route('/api/check-model', methods=['GET'])
def check_model_status():
    """API endpoint to check model compatibility."""
    try:
        is_compatible = check_model_compatibility()
        
        if model is None:
            return jsonify({
                "status": "error",
                "message": "Model not loaded"
            }), 500
        
        return jsonify({
            "status": "success" if is_compatible else "warning",
            "model_loaded": model is not None,
            "vectorizers_loaded": vectorizers is not None,
            "species_list": species_list if species_list else [],
            "compatible": is_compatible,
            "expected_features": len(vectorizers['kmer_4'].vocabulary_) + 24 if vectorizers and 'kmer_4' in vectorizers else 0,
            "model_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else 0
        })
    except Exception as e:
        logger.error(f"Error checking model: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/send-email', methods=['POST'])
def send_email():
    try:
        # Get form data
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"})
            
        name = data.get('name')
        email = data.get('email')
        subject = data.get('subject')
        user_message = data.get('message')
        
        # Validate required fields
        if not all([name, email, subject, user_message]):
            return jsonify({"success": False, "error": "Missing required fields"})
        
        # Email configuration
        sender_email = os.environ.get('EMAIL_HOST_USER', "learnsaxon@gmail.com")
        receiver_email = "learnsaxon@gmail.com"
        password = os.environ.get('EMAIL_HOST_PASSWORD', "maqgsbejobigahbo")
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = f"🧬 DNA Pattern Sequence Finder - {subject}"
        msg['Reply-To'] = email
        
        # Plain text version
        text = f"""
╔══════════════════════════════════════════════════════════════╗
║                🧬 DNA Pattern Sequence Finder                 ║
║                      Contact Form Message                      ║
╚══════════════════════════════════════════════════════════════╝

From: {name} <{email}>
Date: {datetime.now().strftime('%Y-%m-%d at %I:%M %p')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Subject: {subject}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{user_message}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This message was sent through the DNA Pattern Sequence Finder website.
Reply directly to: {email}
        """
        
        # HTML version - Beautiful and professional
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>New Contact Form Submission</title>
</head>
<body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f5f7fa;">
    <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff;">
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; border-radius: 8px 8px 0 0;">
            <div style="font-size: 48px; margin-bottom: 10px;">🧬</div>
            <h1 style="color: white; margin: 0; font-size: 28px; font-weight: 600;">DNA Pattern Sequence Finder</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 16px;">New Contact Form Submission</p>
        </div>
        
        <!-- Content -->
        <div style="padding: 40px; background-color: #ffffff;">
            <!-- From Section -->
            <div style="background-color: #f8f9ff; padding: 20px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #667eea;">
                <div style="font-size: 12px; color: #667eea; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; font-weight: 600;">From</div>
                <div style="font-size: 18px; color: #2d3748; font-weight: 500;">{name}</div>
                <div style="font-size: 14px; color: #718096;">{email}</div>
            </div>
            
            <!-- Subject Section -->
            <div style="background-color: #f8f9ff; padding: 20px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #764ba2;">
                <div style="font-size: 12px; color: #764ba2; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; font-weight: 600;">Subject</div>
                <div style="font-size: 18px; color: #2d3748; font-weight: 500;">{subject}</div>
            </div>
            
            <!-- Message Section -->
            <div style="background-color: #f8f9ff; padding: 20px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #48bb78;">
                <div style="font-size: 12px; color: #48bb78; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; font-weight: 600;">Message</div>
                <div style="font-size: 16px; color: #2d3748; line-height: 1.6; white-space: pre-wrap;">{user_message}</div>
            </div>
            
            <!-- Action Button -->
            <div style="text-align: center; margin: 30px 0;">
                <a href="mailto:{email}" style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 30px; text-decoration: none; border-radius: 25px; font-weight: 500; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);">
                    Reply to {name}
                </a>
            </div>
        </div>
        
        <!-- Footer -->
        <div style="background-color: #f7fafc; padding: 30px; text-align: center; border-radius: 0 0 8px 8px; border-top: 1px solid #e2e8f0;">
            <div style="font-size: 24px; margin-bottom: 10px;">🧬</div>
            <p style="color: #718096; margin: 0; font-size: 14px;">This message was sent from the DNA Pattern Sequence Finder contact form</p>
            <p style="color: #a0aec0; margin: 5px 0 0 0; font-size: 12px;">Sent on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
    </div>
</body>
</html>
        """
        
        # Attach both plain text and HTML versions
        part1 = MIMEText(text, 'plain')
        part2 = MIMEText(html, 'html')
        
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email
        server = smtplib.SMTP(os.environ.get('EMAIL_HOST', 'smtp.gmail.com'), int(os.environ.get('EMAIL_PORT', 587)))
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        
        print("Email sent successfully!")  # Debug print
        response = jsonify({"success": True, "message": "Email sent successfully"})
        print(f"Response: {response.get_json()}")  # Debug print
        return response
        
    except Exception as e:
        print(f"Error in send_email: {e}")  # Debug print
        return jsonify({"success": False, "error": str(e)})
# Cleanup task
@app.before_request
def before_request():
    """Run cleanup tasks before each request."""
    # Clean up old files (only run occasionally to avoid performance impact)
    if np.random.random() < 0.01:  # 1% chance
        cleanup_old_files()

if __name__ == '__main__':
    app.run(debug=True)