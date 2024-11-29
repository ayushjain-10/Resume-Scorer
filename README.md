Project Overview

This project implements two key components:

Category Classification: Fine-tuning a BERT model for classifying text into predefined categories.
Resume Scoring: Using CNN and VGG architectures to predict resume scores based on feature inputs.

Dataset description:
https://huggingface.co/datasets/ahmedheakl/resume-atlas

Table of Contents

Environment Setup
Dataset Preparation
Model Training
Model Evaluation
Running the Code

project-name/
│
├── data/
│   ├── Preprocessed_Data.csv
│   ├── resume_data.csv
│
├── checkpoints/
│
├── train_bert.py
├── evaluate_bert.py
├── train_cnn.py
├── evaluate_cnn.py
├── predict.py
├── requirements.txt
└── README.md










