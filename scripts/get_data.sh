#!/usr/bin/env bash
# Downloads the four public datasets (all CC BY 4.0) from GitHub mirrors into data/raw.
# Originals: IEEE DataPort (Italian), UCI ML Repository (Sakar 2018, Oxford, spiral tablet).
set -e
mkdir -p data/raw && cd data/raw
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/njayem/italianpddataset italian            # ~1.4 GB of WAV
git clone --depth 1 https://github.com/SudeshnaDas246/Speech-Features-based-PD-Classification-using-Combined-SMOTE-ENN-and-Binary-Machine-Learning sakar3
git clone --depth 1 https://github.com/pranjalmedhi/Ensemble_Technique_Predicting_Parkinson-Disease sakar2
git clone --depth 1 https://github.com/shiwalisingh0/Parkinson-disease-detection- spiral_repo
