# Large Language Model - Next Word Prediction
## Overview
Self-created and trained transformer model for next-word (sub-word) prediction, inspired by smartphone keyboard word suggestions. Leveraging knowledge from recent research papers, including *"The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits,"* the model can be initialized with ternary or binary quantization techniques.  
The project aims to demonstrate the practical application of *BitNets* and assess their viability for achieving good accuracy in smaller-scale transformer models, serving as an educational resource for exploring advanced **natural language processing** techniques.

## Installation
`pip install git+https://github.com/majewski00/LLM-Word-Prediction.git` 
  
## Key Features
* **Ternary Quantization:** Incorporates (slightly modified) ternary and binary quantization methods to optimize model performance, as described in recent research papers.
* **Flexibility:** The model can run without any quantization and is compatible with half precision training.
* **Tokenizer:** Utilizes the SentencePieceProcessor tokenizer to handle sub-words efficiently.
* **Model Architecture:** Implements RoPE (Rotary Positional Embedding) and RSMNorm (Root Mean Square Normalization) for improved model performance.
* **Word Suggestion Wrapper:** Includes a WordSuggestionWrapper that wraps the LLM instance to generate next-word prediction and probabilities. This wrapper incorporates a modified *beam search* algorithm for enhanced prediction accuracy.

<br>
<br> 

## Experiments
During the development of this project, several experiments were conducted to assess the performance and capabilities of the model.
### Training
The model was trained on multiple datasets using the HuggingFace datasets library. Successful training lasted for around 24 hours and were conducted using cloud GPU services. Despite being relatively small in size, with approximately *100M parameters*, the model exhibited promising results during training.  
### Objectives
The primary objective of these experiments was to implement and evaluate the usability of ternary quantization. In comparison to tests without quantization, the results showed similar perplexity levels, although the quantized model took longer to achieve them. Additionally, the importance of the *straight-through estimator* ***(STE)*** when applying quantization was emphasized. Furthermore, the experiments aimed to demonstrate the model's ability to generate text with sub-word compatibility.
### Generation Results
```
Input text: This project allows me to ga 
1.  gain      Score: 0.59
2.  gather    Score: 0.26
3.  gate      Score: 0.08  
4.  game      Score: 0.04  
5.  ...       Score: 0.03
```
```
Input text: She glanced at her watch and realized she was running late for her
1.  appointment    Score: 0.39
2.  birthday       Score: 0.25
3.  interview      Score: 0.13
4.  meeting        Score: 0.12
5.  plane          Score: 0.06
6.  ...            Score: 0.05
```
### Observations
The experiments yielded relatively accurate predictions, demonstrating the model's potential despite its brief training period. While the predictions may not be as precise as desired due to the short duration of training, the results are promising and suggest opportunities for further refinement and optimization of the model.
