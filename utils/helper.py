import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

import re
import string

from transformers import AutoTokenizer, GenerationConfig
from utils import MODEL_NAME, DEVICE

import gc
import torch
import numpy as np

## For Data Preparation
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenized_function(example):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    start_prompt = "According to the following question:\n\n"
    end_prompt = "\n\nAnswer:\n\n"
    
    prompt = [start_prompt + question + end_prompt for question in example["question"]]
    
    example['input_ids'] = tokenizer(prompt, truncation=True, padding="max_length", return_tensors="pt").input_ids
    example['labels'] = tokenizer(example['answer'], truncation=True, padding="max_length", return_tensors="pt").input_ids
    
    return example


## For Generating Response
def full_prompt(input_text: str):
    start_prompt = "According to the following question:\n\n"
    end_prompt = "\nAnswer:\n\n"
    
    return start_prompt + input_text + end_prompt

def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    tokenized_prompt = tokenizer(full_prompt(prompt), return_tensors="pt").input_ids.to(DEVICE)
    model_output = model.generate(tokenized_prompt, generation_config=GenerationConfig(max_new_tokens=max_new_tokens))[0]
    final_output = tokenizer.decode(model_output, skip_special_tokens=True)
    clear_gpu_memory()
    return final_output

## For comparing original answers and generated answers
def cosine_similarity(vecA, vecB):
    dot_product = np.dot(vecA, vecB)
    norm_a = np.linalg.norm(vecA)
    norm_b = np.linalg.norm(vecB)
    return dot_product / (norm_a * norm_b)

## For clearing GPU memory
def clear_gpu_memory(debug=False):
    """
    Clears GPU memory on all devices and optionally provides debugging information about memory usage.
    
    Args:
    debug (bool): If True, print memory stats before and after cleanup.
    """
    if debug:
        print("Before cleanup:")
        print(f"Allocated: {torch.cuda.memory_allocated()} bytes")
        print(f"Reserved:  {torch.cuda.memory_reserved()} bytes")

    # Collect garbage to potentially free up memory references
    gc.collect()
    
    # Clear PyTorch's CUDA memory cache
    torch.cuda.empty_cache()
    
    if debug:
        print("After cleanup:")
        print(f"Allocated: {torch.cuda.memory_allocated()} bytes")
        print(f"Reserved:  {torch.cuda.memory_reserved()} bytes")