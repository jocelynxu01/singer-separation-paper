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
| Clean Embedding | Trained on high-quality filtered data | [link](https://drive.google.com/file/d/1DTzLtKgtxBdwuOD5D4K8b9Ugj8PSN-EA/view?usp=drive_link) |
| Noisy Embedding | Trained on unfiltered data | [link](https://drive.google.com/file/d/1LXj0Af3beXb4sgcQL4iJ6QuitASixRiG/view?usp=drive_link) |

### Separation Models

| Model | λ | Link |
|------|----|------|
| FiLM | – | [link](https://drive.google.com/drive/folders/1jrpADqXGYaeFwhDy-35bcHKq5MUHaidy?usp=drive_link) |
| FiLM | 0.05 | [link](https://drive.google.com/drive/folders/18ZQs5OJDzcSDQviJdjKW45OxjJHtUTDo?usp=drive_link) |
| FiLM | 0.1 | [link](https://drive.google.com/drive/folders/1Ie_yzfhz0VCFS4GC136Rq3YXnqYPHD7_?usp=drive_link) |
| FiLM | 0.2 | [link](https://drive.google.com/drive/folders/1pzFy2GGWJNywCIIafwrj-tH8O7bBkG94?usp=drive_link) |
| Concatenation | – | [link](https://drive.google.com/drive/folders/183kVl-I_jRTlwCZC9aG3cGQkDdzIwPsT?usp=drive_link) |
| Concatenation | 0.05 | [link](https://drive.google.com/drive/folders/1HnImboaTn7XREHJZ2fxHBnrrGY9Pf5EH?usp=drive_link) |
| Concatenation | 0.1 | [link](https://drive.google.com/drive/folders/1ovJF41lq71FhCwD44qRHkrAr_bX3_ki3?usp=drive_link) |
| Concatenation | 0.2 | [link](https://drive.google.com/drive/folders/1nQxKw8JM1teJa5ZS2WEw4oTFoAkes57F?usp=drive_link) |


## Acknowledgment

This repository includes a modified version of Open-Unmix.
Original implementation: https://github.com/sigsep/open-unmix-pytorch

The `open-unmix-pytorch/` directory is a modified version of Open-Unmix. The original README is preserved inside that directory, while the paper-specific training scripts are in `open-unmix-pytorch/scripts/`.
This directory is adapted from Open-Unmix for singer-informed vocal source separation.

Main modifications:
- Added singer-conditioned separation models.
- Added concatenation and FiLM conditioning variants.
- Added SI-SDR training scripts.
- Added embedding precomputation scripts.