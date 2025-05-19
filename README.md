### 1. Dataset

#### Dataset download

We with reference to the datasets provided by scDiffusion, PBMC and Muris can be found in scDiffusion ([PBMC and Muris](https://github.com/EperLuo/scDiffusion?tab=readme-ov-file))
Brainsmall dataset referenced in [ACTIVA](https://zenodo.org/records/5842658).
### 2. Autoencoder Training

```bash
cd VAE
#training pbmc: 
python VAE_train.py --data_dir '../data/pbmc68k/data/pbmc68k/filtered_matrices_mex/68k_pbmc_barcodes_annotation.tsv' --num_genes 17789 --save_dir '../data/pbmc' --max_steps 200000
#training muris: 
python VAE_train.py --data_dir '../data/muris/data/tabula_muris/all.h5ad' --num_genes 18996 --save_dir '../data/muris' --max_steps 200000
#training brainsmall: 
python VAE_train.py --data_dir '../data/1M_neurons_neuron20k.h5' --num_genes 17970 --save_dir '../data/brainsmall' --max_steps 200000
```
### 3. Self-supervised model Training
For the pre-guidance mechanism, we use the Masked Autoencoder Vision Transformer ([mae-vit](https://github.com/facebookresearch/mae)) as a learner of external 
gene representations, and by randomly masking and training the model to reconstruct the masked portions, the 
self-supervised model is able to efficiently learn higher-quality gene representations.

### 4. Diffusion Training

```bash
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

### 5. cell samples

```bash
torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 generate.py --ckpt 'exps/linear-mae-b-enc8-pbmc/checkpoints/0600000.pt' --model scDiT-B/4 --num-classes 11 --data_name 'pbmc' --num-fid-samples 3072 --path-type=linear --encoder-depth=8 --projector-embed-dims=768 --per-proc-batch-size=128 --mode=ode --num-steps=50 --heun --cfg-scale=1.0 --guidance-high=1.0
torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 generate.py --ckpt 'exps/linear-mae-b-enc8-muris/checkpoints/0800000.pt' --model scDiT-B/4 --num-classes 12 --data_name 'muris' --num-fid-samples 3072 --path-type=linear --encoder-depth=8 --projector-embed-dims=768 --per-proc-batch-size=128 --mode=ode --num-steps=50 --heun --cfg-scale=1.0 --guidance-high=1.0
torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 generate.py --ckpt 'exps/linear-mae-b-enc8-brainsmall/checkpoints/0800000.pt' --model scDiT-B/4 --num-classes 8 --data_name 'brainsmall' --num-fid-samples 3072 --path-type=linear --encoder-depth=8 --projector-embed-dims=768 --per-proc-batch-size=128 --mode=ode --num-steps=50 --heun --cfg-scale=1.0 --guidance-high=1.0
```

### 6. Acknowledgement
This code is mainly built upon [scDiffusion](https://github.com/EperLuo/scDiffusion?tab=readme-ov-file) and [REPA](https://github.com/sihyun-yu/REPA) repositories.

