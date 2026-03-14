# EvoHash — LLM-guided Evolutionary Discovery of Collision Attacks on Perceptual Hash Functions

Perceptual Hash Functions (PHF), such as PhotoDNA, PDQ, pHash, and NeuralHash, are essential for automated content moderation and digital rights management. Their primary purpose is to map visually similar media to identical or nearly identical digests while ensuring that distinct images remain separated in hash space. However, the discovery of hash collisions—distinct images that yield the same hash value—presents a significant security risk to platforms relying on these systems for identifying illegal content or copyrighted material.

Currently, most collision attacks are manually engineered and struggle to scale across different PHF architectures. This project aims to design and implement EvoHash, a framework that utilizes the OpenEvolve or GigaEvo: An Open Source Optimization Framework Powered By LLMs And Evolution Algorithms evolutionary programming system to automatically discover and optimize complex, multi-scale collision attacks.

## Objective

The primary goal is to develop an automated pipeline that evolves attack code to generate target hash collisions for four widely deployed PHF. The framework should discover strategies that maximize the Success Rate / L2 Distance ratio, achieving high attack success rates while minimizing perceptual distortion.

## Key Research Questions

- Can evolutionary algorithms guided by LLMs automatically discover effective collision attacks without manual engineering?
- How do evolved attacks transfer across different perceptual hashing architectures?
- What attack patterns emerge from the evolutionary process, and can they reveal fundamental vulnerabilities in PHF designs?

## Threat Model & Baselines

We evaluate against four production-grade PHF with their corresponding collision thresholds:

- pHash with threshold p = 12
- PhotoDNA with threshold p = 3855
- NeuralHash with threshold p = 17
- PDQ with threshold p = 92

These thresholds represent the maximum hash distance at which two images are considered perceptually similar.

**Baselines:**
NES, SimBa, ZO-Sign-SGD, NES+HSJA, SimBa+HSJA, ZO-Sign-SGD+HSJA, Prokos, Atkscopes.

## Experimental Setup

**Models:**
- GPT-OSS
- GigaChat-2-Max

**Dataset:**
- 100 Random images pairs from ImageNet Val

## Evaluation Metrics

### Primary Metrics

- **Attack Success Rate (ASR)**
- **Success Rate / L2:** Measures attack efficiency—higher values indicate successful collisions with minimal distortion.
- **L2 Distance:** Average pixel-space distortion between original image and perturbed image.
- **Time:** Average wall-clock time per attack (in seconds).

### Secondary Metrics

- **LPIPS (Learned Perceptual Image Patch Similarity):** Deep perceptual distance
- **Query Efficiency:** Average number of hash queries required per successful collision
- **Transferability:** Success rate of attacks evolved for one PHF when applied to others

## Expected Outcomes

- A practical pipeline for evolutionary generation of attacks against four perceptual hash functions.
- An empirical analysis of attack generalizability strategies, including:
  - Cross-architecture transfer learning
  - Identification of universal vulnerabilities in PHF designs
- Open-source code and reproducible experiments, including:
  - Complete EvoHash framework implementation
  - Pre-evolved attack code for each target PHF
  - Benchmark dataset and evaluation scripts
  - Analysis notebooks demonstrating attack patterns