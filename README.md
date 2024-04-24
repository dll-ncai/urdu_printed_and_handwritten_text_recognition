# Unified Architecture for Urdu Printed and Handwritten Text Recognition
This repository contains the implementation of a unified architecture for Urdu printed and handwritten text recognition. The architecture comprises a custom CNN block with a Transformer encoder for image understanding, coupled with a pre-trained Transformer decoder for Urdu language modeling.

For detailed information about the architecture and its implementation, please refer to the paper:
**[A Unified Architecture for Urdu Printed and Handwritten Text Recognition](https://dl.acm.org/doi/10.1007/978-3-031-41685-9_8)**

## Contents
- `data_uhwr/, data_upti/ and data_urti/`: Directories for datasets used in experiments.
- `vocabs/ved`: Vocabulary file for dataset.
- `Training_conv_transformer_ICDAR.ipynb`: Training script for the architecture.
- `models.py`: Model implementation.
- `README.md`: You are reading this file, which serves as a guide to the repository.

## Requirements
- Python 3.11
- pandas, tqdm, pillow, opencv-python, taco-box
- PyTorch
- Transformers

## Usage
After installing all the required libraries, you just need to execute the 'Training_conv_transformer_ICDAR.ipynb' to train the model.

