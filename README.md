# Singer-Informed Vocal Source Separation

This repository contains the full experimental pipeline for singer-informed vocal source separation, including dataset construction, vocal activity filtering, singer embedding learning, and Open-Unmix-based separation models.

The codebase is organized to support end-to-end reproducibility of the experiments presented in the paper.

---

## Overview

The pipeline consists of the following stages:

1. **Data Preparation**  
   Construct solo and duet mixtures from raw vocal recordings.

2. **Quality Filtering and VAD**  
   Filter recordings using DNSMOS and extract valid vocal segments using energy-based voice activity detection.

3. **Singer Embedding**  
   Train a singer embedding model or use pretrained embeddings.

4. **Data Splitting**  
   Partition data into train, validation, and test sets.

5. **Embedding Precomputation**  
   Precompute singer embeddings for conditioning separation models.

6. **Separation Model Training and Evaluation**  
   Train singer-conditioned Open-Unmix models or evaluate using pretrained checkpoints.

---

## Repository Structure

```text
.
├── data_preparation/             # Dataset construction (solo and duet generation)
├── VAD/                          # DNSMOS filtering and energy-based VAD
├── singer_embedding/             # Singer embedding model and training scripts
├── prepare_trainvalidtest_data/  # Dataset splitting utilities
├── open-unmix-pytorch/scripts    # Modified Open-Unmix implementation

```

---

## Pretrained Models

Pretrained checkpoints are available below. Please download and place them in:

### Singer Embedding Models

| Model | Description | Link |
|------|------------|------|
| Clean Embedding | Trained on high-quality filtered data | [link] |
| Noisy Embedding | Trained on unfiltered data | [link] |

### Separation Models

| Model | Conditioning | λ | Link |
|------|-------------|----|------|
| FiLM | FiLM | – | [link] |
| FiLM | FiLM | 0.05 | [link] |
| FiLM | FiLM | 0.1 | [link] |
| FiLM | FiLM | 0.2 | [link] |
| Concatenation | Concat | – | [link] |
| Concatenation | Concat | 0.05 | [link] |
| Concatenation | Concat | 0.1 | [link] |
| Concatenation | Concat | 0.2 | [link] |