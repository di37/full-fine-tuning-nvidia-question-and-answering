# Full Fine-Tuning of Flan-t5-base Model

This project demonstrates the process of fully fine-tuning the Flan-T5-Base model for the NVIDIA question-answering task. The main objective of this project is to provide beginners with hands-on experience in fine-tuning a large language model, rather than achieving a perfect model.

## Environment Setup

1. Setup Conda environment
```bash
conda create -n fine-tuning-q-and-a python=3.11
conda activate fine-tuning-q-and-a
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

## Overview

- The Flan-T5-Base model, a powerful language model, is used as the base model for fine-tuning.
- The model is fine-tuned specifically for the NVIDIA question-answering task.
- The project serves as an educational resource for beginners to understand and practice the fine-tuning process.

## Machine Learning Process

- The dataset used for fine-tuning the model consists of question-answer pairs related to NVIDIA taken from [Kaggle](https://www.kaggle.com/datasets/gondimalladeepesh/nvidia-documentation-question-and-answer-pairs). The dataset is prepared appropriately for training the model followed by performing tokenization on the whole dataset. Check `01_Data_Preparation.ipynb` and `02_Data_Tokenization.ipynb`for running the code implementation.
- Model training was done using the tokenized dataset in `03_Model_Training.ipynb` notebook.
- Once training was done, model was then evaluated based on performance metric - evaluation loss and also based on qualitative analysis in the notebooks - `04_Model_Evaluation_Performance_Metric.ipynb` and `05_Model_Evaluation_Qualitative_Analysis.ipynb`.
- For nice readibility of the code, all of the functions are included in `helper.py` and the environment variables are loaded `constants.py ` in `utils` folder. Check these files for more details understanding of the functions.

## Reference 
Special thanks to Eng. Omar M. Atef for creating course on Udemy: [LLMs Workshop: Practical Exercises of Large Language Models](https://www.udemy.com/course/llms-workshop-practical-exercises-of-large-language-models). Full video tutorial can be found in Section 2 of this course.
