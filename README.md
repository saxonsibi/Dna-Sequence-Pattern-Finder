# 🧬 DNA Sequence Pattern Finder

<p align="center">
  <img src="https://github.com/saxonsibi/Dna-Sequence-Pattern-Finder/assets/140614065/aa4df8bb-be5e-4e57-9c06-6dc3d54bee43" alt="Animated DNA double helix" width="180"/>
</p>

> **Advanced machine learning meets modern bioinformatics.**  
> Identify species, discover DNA motifs, and analyze genetic mutations—fast, accurately, and at scale.

<div align="center">

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](docs/CONTRIBUTING.md)

</div>

---

## 🚀 Overview

**DNA Sequence Pattern Finder** is a professional-grade web application for rapid DNA sequence analysis, leveraging machine learning to:

- **Identify species** from sequence fragments or full sequences
- **Discover motifs and patterns**, including conserved regions, palindromes, and repeats
- **Analyze mutations** between sequences or against references
- Visualize **GC/AT content**, entropy, CpG ratio, complexity, and more—all in one place

Ideal for biologists, geneticists, and data scientists in research, diagnostics, or education.

---

## ✨ Features at a Glance

- **Accurate ML-based species prediction** for DNA sequences (FASTA/TXT)
- **Batch processing:** Analyze hundreds of sequences at once
- **Dynamic visualizations:** GC content distribution, motif maps, mutation hotspots
- **Comprehensive stats:** Nucleotide frequencies, entropy, CpG, homopolymers, etc.
- **Clean, responsive UI**: Recommended web fonts: `Segoe UI`, `Roboto`, `Arial`, `sans-serif`

---

## 📦 Quickstart

### 1. Installation

```bash
git clone https://github.com/saxonsibi/Dna-Sequence-Pattern-Finder.git
cd Dna-Sequence-Pattern-Finder
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and set any required environment variables.

### 3. Launch the App

```bash
python dna_flask.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 💡 Usage Examples

### Predict Species

- Paste/upload a DNA sequence on the homepage.
- Click **Start Analysis** for fast, detailed predictions.

### Batch Processing

- Upload a `.fasta`, `.fa`, or `.txt` file with many sequences.
- Download results/statistics for all samples.

### Mutation/Pattern Analysis

- Use **Advanced Analysis** for mutation detection or motif discovery against references.

---

## 📚 Documentation & Support

- **In-app help:** Use built-in tooltips and guides.
- **Issues & bugs:** Please report via [GitHub Issues](issues/).
- **Full documentation:** See [`docs/`](docs/) for guides & in-depth details.

---

## 👥 Contribution & Maintenance

- **Maintainer:** [@saxonsibi](https://github.com/saxonsibi)
- **License:** See [LICENSE](LICENSE)
- **Contributions welcome:** @saxonsibi,@Nandana KM

---

**Font Recommendations for the Web UI:**  
`font-family: 'Segoe UI', 'Roboto', 'Arial', 'sans-serif';`

---

<p align="center"><sub>For advanced and API use, see the <a href="docs/">docs/</a> folder.</sub></p>
