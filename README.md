# ESCFD: Probabilistic Flow Diffusion Model for Accelerated High-Quality Single-Cell RNA-seq Data Synthesis
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15515721.svg)](https://doi.org/10.5281/zenodo.15515721)
[![Paper](https://img.shields.io/badge/arXiv%20paper-2410.06940-b31b1b.svg)](https://dl.acm.org/doi/10.1145/3711896.3736942)
### 1. Environment 
The environment settings refer to [scDiffusion](https://github.com/EperLuo/scDiffusion?tab=readme-ov-file) and [mae-vit](https://github.com/facebookresearch/mae)


```bash
pytorch                   1.13.0  
numpy                     1.23.4  
anndata                   0.8.0  
scanpy                    1.9.1  
scikit-learn              1.2.2  
blobfile                  2.0.0  
pandas                    1.5.1  
celltypist                1.3.0  
imbalanced-learn          0.11.0  
mpi4py                    3.1.4  
```
You need to change (img_size = 8) and (in_chans = 2) in class VisionTransformer in (.conda/envs/escfd/lib/python3.10/site-packages/timm/models/vision_transformer.py)

### 2. Dataset

We with reference to the datasets provided by scDiffusion, PBMC and Muris can be found in scDiffusion ([PBMC and Muris](https://github.com/EperLuo/scDiffusion?tab=readme-ov-file)). Brainsmall dataset referenced in [ACTIVA](https://zenodo.org/records/5842658).

### 3. Autoencoder Training

```bash
cd VAE
#training pbmc: 
python VAE_train.py --data_dir '../data/pbmc68k/data/pbmc68k/filtered_matrices_mex/68k_pbmc_barcodes_annotation.tsv' --num_genes 17789 --save_dir '../data/pbmc' --max_steps 200000
#training muris: 
python VAE_train.py --data_dir '../data/muris/data/tabula_muris/all.h5ad' --num_genes 18996 --save_dir '../data/muris' --max_steps 200000
#training brainsmall: 
python VAE_train.py --data_dir '../data/1M_neurons_neuron20k.h5' --num_genes 17970 --save_dir '../data/brainsmall' --max_steps 200000
```
### 4. Self-supervised model Training
For the pre-guidance mechanism, we use the Masked Autoencoder Vision Transformer ([mae-vit](https://github.com/facebookresearch/mae)) as a learner of external gene representations, and by randomly masking and training the model to reconstruct the masked portions, the self-supervised model is able to efficiently learn higher-quality gene representations.

```bash
cd ../mae

python main_pretrain.py \
    --data_dir 'data/pbmc68k/data/pbmc68k/filtered_matrices_mex/hg19' \
    --vae_path 'data/pbmc_AE/model_seed=0_step=199999.pt' \
    --data_name 'pbmc' \
    --batch_size 128 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 1600 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05
```
The weights of the self-supervised encoder are placed in the ‘ckpt’ folder.

### 5. Diffusion Training

```bash
cd ../

accelerate launch train.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="scDiT-B/4" \
  --enc-type="mae-vit-b" \
  --batch-size=32 \
  --proj-coeff=0.5 \
  --encoder-depth=8 \
  --num-classes=11 \
  --output-dir="exps" \
  --exp-name="linear-mae-b-enc8-pbmc" \
  --max-train-steps=800000 \
  --data-dir="data/pbmc68k/data/pbmc68k/filtered_matrices_mex/hg19" \
  --vae_path="data/pbmc_AE/model_seed=0_step=199999.pt"
```
- `--proj-coeff`: Any values larger than 0
- `--encoder-depth`: Any values between 1 to the depth of the model
- `--output-dir`: the path to save checkpoints and logs
- `--enc-type`:  any self-supervised model

Diffusion weights are placed in the ‘exps’ folder.
### 6. cell samples

```bash
torchrun --nnodes=1 --nproc_per_node=1  generate.py --ckpt 'exps/linear-mae-b-enc8-pbmc/checkpoints/0600000.pt' --model scDiT-B/4 --num-classes 11 --data_name 'pbmc' --num-fid-samples 3072 --path-type=linear --encoder-depth=8 --projector-embed-dims=768 --per-proc-batch-size=128 --mode=ode --num-steps=50 --heun --cfg-scale=1.0 --guidance-high=1.0
torchrun --nnodes=1 --nproc_per_node=1  generate.py --ckpt 'exps/linear-mae-b-enc8-muris/checkpoints/0800000.pt' --model scDiT-B/4 --num-classes 12 --data_name 'muris' --num-fid-samples 3072 --path-type=linear --encoder-depth=8 --projector-embed-dims=768 --per-proc-batch-size=128 --mode=ode --num-steps=50 --heun --cfg-scale=1.0 --guidance-high=1.0
torchrun --nnodes=1 --nproc_per_node=1  generate.py --ckpt 'exps/linear-mae-b-enc8-brainsmall/checkpoints/0800000.pt' --model scDiT-B/4 --num-classes 8 --data_name 'brainsmall' --num-fid-samples 3072 --path-type=linear --encoder-depth=8 --projector-embed-dims=768 --per-proc-batch-size=128 --mode=ode --num-steps=50 --heun --cfg-scale=1.0 --guidance-high=1.0
```

### 7. cell samples output and weight
Our weight and datasets is comming soon.


### 8. Acknowledgement
This code is mainly built upon [scDiffusion](https://github.com/EperLuo/scDiffusion?tab=readme-ov-file), [mae-vit](https://github.com/facebookresearch/mae) and [REPA](https://github.com/sihyun-yu/REPA) repositories.

