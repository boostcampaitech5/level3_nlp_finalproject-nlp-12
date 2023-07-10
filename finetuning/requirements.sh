#!/bin/bash
### install requirements for pstage3 baseline
# pip requirements
pip install --upgrade pip
pip install torch
pip install datasets
pip install tqdm
pip install pandas
pip install omegaconf
pip install wandb
pip install huggingface-hub
pip install scipy

pip install fire
pip install gradio
pip install langchain 
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git 
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
pip install -q datasets