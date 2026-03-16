# AminSFlow

## Installation

```
# clone project
git clone --recursive https://github.com/alideroo/AminSFlow.git
cd AminSFlow
```

```
# create conda virtual environment
conda env create -f env.yml
conda activate AminSFlow
pip install -r requirements.txt
```

## Structure-based protein sequence design

## Data

```
# Download the preproceesd CATH datasets
# CATH 4.2 dataset provided by Generative Models for Graph-Based Protein Design (https://papers.nips.cc/paper/2019/hash/f3a4ff4839c56a5f460c88cce3666a2b-Abstract.html)
# CATH 4.3 dataset provided by Learning inverse folding from millions of predicted structures
bash scripts/download_cath.sh

```
Go check configs/datamodule/cath_4.*.yaml and set data_dir to the path of the downloaded CATH data.

## Training 

```
model=bridge_if_esm1b_650m_pifold
exp=fixedbb/${model}
dataset=cath_4.2
name=fixedbb/${dataset}/${model}

python ./train.py \
    experiment=${exp} datamodule=${dataset} name=${name} \
    task.generator.diffusion_steps=steps \
    logger=wandb trainer=ddp_fp16
```

## Evaluation/inference on valid/test datasets

```
dataset=cath_4.2
name=fixedbb/${dataset}/aminfold_if_esm1b_650m_pifold
exp_path=logs/${name}

python ./test.py \                                                                 
    experiment_path=${exp_path} \
    data_split=test ckpt_path=best.ckpt mode=predict
```
