Project Overview

This project implements two key components:

Category Classification: Fine-tuning a BERT model for classifying text into predefined categories.

Resume Scoring: Using CNN and VGG architectures to predict resume scores based on feature inputs.

Dataset description:
https://huggingface.co/datasets/ahmedheakl/resume-atlas

Table of Contents

1. Environment Setup

2. Dataset Preparation

3. Model Training

4. Model Evaluation

5. Running the Code

1.To run this project, ensure your system meets the following requirements and dependencies.
torch==1.11.0
transformers==4.28.1
scikit-learn==1.0.2
pandas==1.3.3
numpy==1.21.2
matplotlib==3.4.3
tqdm==4.62.3

2.Dataset Preparation 
Please see the Preprocessed_Data.csv, test_normalized_cv.csv and train_normalized_cv.csv

To run code for Category Classification with BERT, run train_bert.py

To run code for Resume Scoring with CNN/VGG, run shallow_cnn.py










