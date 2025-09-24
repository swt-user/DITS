# DITS: Efficient Multi-Agent System Training with Data Influence-Oriented Tree Search

**DITS** a novel framework that incorporates influence scores to guide both tree search and data selection. By leveraging influence scores, we effectively identify the most impactful data for system improvement, thereby enhancing model performance. Furthermore, we derive influence score estimation methods tailored for non-differentiable metrics, significantly reducing computational overhead by utilizing inference computations.




## 🛠 Installation

DITS requires two conda environments: one for vLLM deployment and another for training, both using Python 3.11. Follow these steps to set up your environments:

### vLLM Environment

```bash
conda create -n DITS-vllm python=3.11
# Install NVCC for CUDA
conda install nvidia/label/cuda-12.1.0::cuda-nvcc
# Install PyTorch 2.3.1 (required by VLLM==0.6.1.post1)
conda install pytorch=2.3.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# Install VLLM 0.6.1.post1
pip install vllm==0.6.1.post1
```

### Training Environment

```bash
conda create -n DITS-train python=3.11
# Install NVCC for CUDA
conda install nvidia/label/cuda-12.1.0::cuda-nvcc
# Install PyTorch 2.3.1 (required by VLLM==0.6.1.post1)
conda install pytorch=2.3.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# Install Alignment Handbook 
cd alignment-handbook
pip install -e .
cd ..
# Install Dependencies
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
```


## 🏃‍♂️ How to Run

### Prepare datasets


To ensure proper execution of the code, the following path adjustments are required:  

1. **File Path Replacement**  
   - Replace all instances of `YOUR_HOME_PATH_HERE` and `YOUR_FILE_PATH_HERE` in the codebase with the corresponding local file paths.  

2. **Model Path Specification**  
   - Substitute `YOUR_MODEL_PATH_HERE` with the actual path to the downloaded model.  

These modifications are necessary for the code to function as intended.  


Run the combined DITS-iSFT-DPO setting with hotpotqa dataset:

```bash
export INITIAL_MODEL_PATH="YOUR_MODEL_PAHT_HERE"
export DATASET='hotpot_qa'
export TOKENIZERS_PARALLELISM=false

MKL_THREADING_LAYER=GNU OUTLINES_CACHE_DIR='./outlines/hotpot_qa' python sft_dpo_script_DI.py \
    --train_config_path train/sft_dpo_recipes/hotpot_qa_DI.yaml \
    --vllm_env DITS-vllm \
    --alignment_env DITS-train 
```

## 📊 Inference and Evaluation

### Inference Script

To get inference results for all models on a specified test set:

```bash
OUTLINES_CACHE_DIR=./outlines/${dataset_type} 

python inference_script.py \
    --model_root_path YOUT_HOME_PATH_HERE/checkpoints/${dataset_type}_sft_dpo_DI/ \
    --tokenizer_path YOUR_MODEL_PATH_HERE \
    --device 0 \
    --port 8000 \
    --dataset_type ${dataset_type} \
    --num_thread 128 \
    --num_gpu 4 \
    --output_root_path ${output_file} \
    --vllm_env DITS-vllm \
``` 

