# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models.scDiT import scDiT_models

from tqdm import tqdm
import os

import numpy as np
import math
import argparse
from samplers import euler_sampler, euler_maruyama_sampler
from utils import load_legacy_checkpoints, download_model

def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    torch.set_grad_enabled(False)

    # Setup device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.global_seed
    torch.manual_seed(seed)

    
    # Load model:
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 8 #latent_size 是输入尺寸的1/8
    model = scDiT_models[args.model](
        input_size=latent_size,#模型输入是VAE的潜在空间。
        num_classes=args.num_classes,
        use_cfg = True,
        z_dims = [int(z_dim) for z_dim in args.projector_embed_dims.split(',')],
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    ).to(device)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = torch.load(ckpt_path, map_location=device)['ema']
    if args.legacy:
        state_dict = load_legacy_checkpoints(
            state_dict=state_dict, encoder_depth=args.encoder_depth
            )
    model.load_state_dict(state_dict)
    model.eval()  # important!

    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.resolution}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}-{args.mode}-{args.data_name}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    os.makedirs(sample_folder_dir, exist_ok=True)
    print(f"Saving .png samples at {sample_folder_dir}")

    # Figure out how many samples we need to generate and how many iterations we need to run:
    n = args.per_proc_batch_size
    total_samples = args.num_fid_samples
    iterations = int(total_samples // n)
    pbar = tqdm(range(iterations))
    total = 0
    #########sample##########
    for c in range(args.num_classes+1):
        all_samples = []

        print(c)
        import time
        total_start_time = time.time()
        for _ in pbar:
            z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
            y = torch.full((n,), c, device=device)


            sampling_kwargs = dict(
                model=model,
                latents=z,
                y=y,
                num_classes=args.num_classes,
                num_steps=args.num_steps,
                heun=args.heun,
                cfg_scale=args.cfg_scale,
                guidance_low=args.guidance_low,
                guidance_high=args.guidance_high,
                path_type=args.path_type,
            )
    

            with torch.no_grad():

                if args.mode == "sde":
                    samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
                elif args.mode == "ode":
                    samples = euler_sampler(**sampling_kwargs).to(torch.float32)
                else:
                    raise NotImplementedError()
    

                samples = samples.reshape(samples.shape[0], args.resolution*2)
    
                all_samples.append(samples.cpu().numpy())
    

            total += n
        total_time = time.time() - total_start_time
        print(f'Total time taken: {total_time:.2f} seconds.')

        all_samples = np.concatenate(all_samples, axis=0)
    

        npz_path = os.path.join(f"{sample_folder_dir}", f"{args.data_name}{c}.npz")
        np.savez(npz_path, samples=all_samples)
        print(f"Saved .npz file to {npz_path} [shape={all_samples.shape}].")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--ckpt", type=str, default='exps/linear-dinov2-s-enc8/checkpoints/0020000.pt', help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")

    # model
    parser.add_argument("--model", type=str, choices=list(scDiT_models.keys()), default="scDiT-B/4")
    parser.add_argument("--num-classes", type=int, default=11)#muris 12
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[64], default=64)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--data_name", type=str, choices=["pbmc", "muris", "brainsmall"])
    # vae
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")

    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=1000)
    parser.add_argument("--num-fid-samples", type=int, default=3000)

    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="ode")
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--projector-embed-dims", type=str, default="768")#,1024
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False) # only for ode
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)

    # will be deprecated
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False) # only for ode


    args = parser.parse_args()
    main(args)
