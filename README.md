# Fine-Tuning Llama-2 with QLoRA

Llama-2 is a powerful language model that can now be fine-tuned on your own data with ease, thanks to the optimized script provided here. This script allows for efficient fine-tuning on both single and multi-GPU setups, and it even enables training the massive 70B model on a single A100 GPU by utilizing 4-bit precision.

## Introduction

This repository contains an optimized implementation for fine-tuning the Llama-2 model using QLoRA (Quantization-Aware Layer-wise Rate Allocation). The code has been refactored and organized to achieve better performance while training the model on custom datasets.

## Installation

1. Clone this repository to your local machine:

```
git clone https://github.com/mert-delibalta/llama2-fine-tune-qlora.git
cd llama2-fine-tune-qlora
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

To fine-tune Llama-2 on your data, follow these simple steps:

1. Prepare your dataset in the required format.
2. Modify the script arguments (if needed) to set your desired options.
3. Run the fine-tuning script:

```
python train.py
```

## Benefits of 4-bit Precision

By enabling 4-bit precision (activated by default), you can train the massive 70B model even on a single A100 GPU, making it accessible to a broader range of users.

## Options

The fine-tuning script supports various command-line options to customize the training process. You can access the model and training data on Hugging Face. Some of the essential options are:

- `--model_name`: Specify the model to be fine-tuned. Default: "meta-llama/Llama-2-7b-hf".
- `--dataset_name`: Select the preference dataset to use. Default: "timdettmers/openassistant-guanaco".
- `--use_4bit`: Activate 4-bit precision base model loading. Default: True.
- `--use_nested_quant`: Activate nested quantization for 4-bit base models. Default: False.
- ... (other options can be found in the code)

## Requirements

The optimized script is compatible with the following library versions:

- Python >= 3.6
- accelerate == 0.21.0
- peft == 0.4.0
- bitsandbytes == 0.40.2
- transformers == 4.31.0
- trl == 0.4.7

## License

This code is licensed under the Apache License, Version 2.0.
