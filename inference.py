import torch

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    raise Exception("unable to initialize CUDA")

import os
import gc
import math
import datetime
import argparse
from time import perf_counter
from tqdm import tqdm
from safetensors.torch import load_file
from torchvision.transforms import v2

from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.modules.model import WanModel
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import cache_video

import decord
decord.bridge.set_bridge('torch')
from peft import LoraConfig, set_peft_model_state_dict
import traceback

from utils.residual_head import ResidualHead, residual_in_channels

def convert_sd(sd):
    converted_sd = {}
    for key in sd.keys():
        renamed_key = key
        
        
        if ".reshape_weight" in key:
            continue
        
        if "diffusion_model." in renamed_key:
            renamed_key = renamed_key.replace("diffusion_model.", "")
        
        converted_sd[renamed_key] = sd[key].to(torch.float32)
    
    return converted_sd


def _apply_residual_head(residual_head, args, noise_pred_uncond, control_latents, timestep, context, seq_len):
    if residual_head is None:
        return noise_pred_uncond

    if args.residual_input == "pred":
        res_in = noise_pred_uncond
    elif args.residual_input == "control":
        if control_latents is None:
            res_in = torch.zeros_like(noise_pred_uncond)
        else:
            res_in = control_latents.to(dtype=noise_pred_uncond.dtype, device=noise_pred_uncond.device)
    elif args.residual_input == "pred+control":
        if control_latents is None:
            control_stack = torch.zeros_like(noise_pred_uncond)
        else:
            control_stack = control_latents.to(dtype=noise_pred_uncond.dtype, device=noise_pred_uncond.device)
        if noise_pred_uncond.dim() == 4:
            res_in = torch.cat([noise_pred_uncond, control_stack], dim=0)
        else:
            res_in = torch.cat([noise_pred_uncond, control_stack], dim=1)
    else:
        raise ValueError(f"Unknown residual_input: {args.residual_input}")

    delta = residual_head(
        res_in.to(dtype=torch.bfloat16, device=noise_pred_uncond.device),
        t=timestep,
        context=context,
        seq_len=seq_len,
    ).to(dtype=noise_pred_uncond.dtype)
    return noise_pred_uncond + delta


def generate_one(control_path, real_output_dir, args, device, vae, diffusion_model,
                 scheduler, positive_context, negative_context, residual_head=None):
    start = perf_counter()

    # ===== read the LQ video =====
    vr_lq = decord.VideoReader(control_path) 
    frames = len(vr_lq)

    # ===== preprocess =====
    latent_width  = (args.width  // 16) * 2
    latent_height = (args.height // 16) * 2
    latent_frames = (frames - 1) // 4 + 1

    width  = latent_width  * 8
    height = latent_height * 8
    frames = (latent_frames - 1) * 4 + 1

    seq_len = math.ceil((latent_height / 2) * (latent_width / 2) * latent_frames)

    torch.manual_seed(args.seed)
    latents = torch.randn(16, latent_frames, latent_height, latent_width, device=device)

    lq_pixels = vr_lq[:frames]                      # [F, H, W, C], uint8
    lq_pixels = lq_pixels.movedim(3, 1).unsqueeze(0).contiguous()  # BFCHW

    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=(height, width)),
    ])

   
    lq_pixels = transform(lq_pixels) * 2 - 1
    lq_pixels = torch.clamp(torch.nan_to_num(lq_pixels), min=-1, max=1)



    lq_pixels = lq_pixels[0].movedim(0, 1).to(device)              # CFHW
    control_pixels = lq_pixels

    

    control_latents = vae.encode([control_pixels.to(dtype=torch.bfloat16, device=device)])[0].to(device)
    assert control_latents.shape == latents.shape, f"latent shape mismatch: {control_latents.shape} {latents.shape}"

    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(args.steps, device=device, shift=args.shift)

    # ===== sampling =====
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for t in tqdm(scheduler.timesteps, desc=os.path.basename(control_path)):
            latent_model_input = [torch.cat([latents, control_latents], dim=0)]
            timestep = torch.stack([t])

            noise_pred_cond = diffusion_model(
                latent_model_input, t=timestep, context=positive_context, seq_len=seq_len
            )[0]
            noise_pred_uncond = diffusion_model(
                latent_model_input, t=timestep, context=negative_context, seq_len=seq_len
            )[0]
            noise_pred_uncond = _apply_residual_head(
                residual_head=residual_head,
                args=args,
                noise_pred_uncond=noise_pred_uncond,
                control_latents=control_latents,
                timestep=timestep,
                context=negative_context,
                seq_len=seq_len,
            )
            noise_pred = noise_pred_uncond + args.cfg * (noise_pred_cond - noise_pred_uncond)

            latents = scheduler.step(
                noise_pred.unsqueeze(0), timestep, latents.unsqueeze(0), return_dict=False
            )[0].squeeze(0)

        decoded_video = vae.decode([latents])[0].to(device)
        control_pixels = control_pixels.to(device)

        # side-by-side comparison video
        cat_dim = -2 if args.width > args.height else -1
        decoded_video_all = torch.cat([control_pixels, decoded_video], dim=cat_dim)
    end = perf_counter()
    print(f"Time: {end - start:.6f} seconds")
    # ===== save =====
    stem = os.path.splitext(os.path.basename(control_path))[0]
    save_path_all = os.path.join(real_output_dir, f"{stem}_all.mp4")
    cache_video(
        tensor=decoded_video_all[None], save_file=save_path_all, 
        fps=16, nrow=1,normalize=True, value_range=(-1, 1),
    )
    # === individual videos ===
    save_path_lq = os.path.join(real_output_dir, f"{stem}_lq.mp4")
    cache_video(
        tensor=control_pixels[None],
        save_file=save_path_lq,
        fps=16, nrow=1, normalize=True, value_range=(-1, 1),
    )
    save_path_pred = os.path.join(real_output_dir, f"{stem}_pred.mp4")
    cache_video(
        tensor=decoded_video[None],
        save_file=save_path_pred,
        fps=16, nrow=1, normalize=True, value_range=(-1, 1),
    )


    return save_path_all

@torch.inference_mode()
def main(args):
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    real_output_dir = os.path.join(args.output_dir, date_time)
    os.makedirs(real_output_dir, exist_ok=True)
    
    if args.prompt is not None or args.negative_prompt is not None:
        ckpt_dir = "./models/UMT5/models_t5_umt5-xxl-enc-bf16.pth"
        tk_dir = "./models/UMT5/google/umt5-xxl"
        umt5_model = T5EncoderModel(
            text_len = 512,
            dtype = torch.bfloat16,
            device = device,
            checkpoint_path = ckpt_dir,
            tokenizer_path = tk_dir,
        )
        
        gc.collect()
        torch.cuda.empty_cache()
        
        positive_context = umt5_model([args.prompt], umt5_model.device)[0].to(device=device, dtype=torch.bfloat16)
        negative_context = umt5_model([args.negative_prompt], umt5_model.device)[0].to(device=device, dtype=torch.bfloat16)
        
        del umt5_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        positive_context = [load_file(args.embedding)["context"].to(device=device, dtype=torch.bfloat16)]
        negative_context = [load_file(args.negative_embedding)["context"].to(device=device, dtype=torch.bfloat16)]
    
    vae = WanVAE(vae_pth="./models/vae/Wan2.1_VAE.pth", dtype=torch.bfloat16, device=device)
    diffusion_model = WanModel.from_pretrained("./models/" + args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16).to(device)
    lora_sd = load_file(args.lora)
    # move all tensors to the model device / dtype
    for k, v in list(lora_sd.items()):
        if torch.is_tensor(v):
            lora_sd[k] = v.to(device=device, dtype=torch.bfloat16)
    lora_sd = convert_sd(lora_sd)

    if args.lora_scope == "patch_embedding":
        before = len(lora_sd)
        lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("patch_embedding.")}
        print(f"[ablation] use patch_embedding LoRA only: {len(lora_sd)}/{before} tensors")
    
    if "patch_embedding.lora_A.weight" in lora_sd:
        in_cls = diffusion_model.patch_embedding.__class__ # nn.Conv3d
        old_in_dim = diffusion_model.in_dim # 16
        new_in_dim = lora_sd["patch_embedding.lora_A.weight"].shape[1]
        assert new_in_dim == 32
        
        new_in = in_cls(
            new_in_dim,
            diffusion_model.patch_embedding.out_channels,
            diffusion_model.patch_embedding.kernel_size,
            diffusion_model.patch_embedding.stride,
            diffusion_model.patch_embedding.padding,
        ).to(device=device, dtype=torch.bfloat16)
        
        new_in.weight.zero_()
        new_in.bias.zero_()
        
        new_in.weight[:, :old_in_dim].copy_(diffusion_model.patch_embedding.weight)
        new_in.bias.copy_(diffusion_model.patch_embedding.bias)
        
        diffusion_model.patch_embedding = new_in
        diffusion_model.register_to_config(in_dim=new_in_dim)
    
    if not (args.patch_embedding and os.path.exists(args.patch_embedding)):
        raise FileNotFoundError(f"patch embedding checkpoint not found: {args.patch_embedding}")
    print(f"[load] patch embedding: {args.patch_embedding}")
    pe_sd = torch.load(args.patch_embedding, map_location="cpu")
    with torch.no_grad():
        diffusion_model.patch_embedding.weight.copy_(
            pe_sd["weight"].to(device=device, dtype=torch.bfloat16)
        )
        if pe_sd.get("bias") is not None and diffusion_model.patch_embedding.bias is not None:
            diffusion_model.patch_embedding.bias.copy_(
                pe_sd["bias"].to(device=device, dtype=torch.bfloat16)
            )
    
    lora_params = []
    lora_params.append("patch_embedding")
    
    if args.lora_scope == "all":
        lora_targets = args.lora_targets.replace(" ", "").split(",")
        lora_blocks = args.lora_blocks.replace(" ", "").split(",") if args.lora_blocks is not None else None
        for name, param in diffusion_model.named_parameters():
            if name.endswith(".weight"):
                for target in lora_targets:
                    if target in name and ".norm_" not in name:
                        if lora_blocks is None:
                            lora_params.append(name.replace(".weight", ""))
                        else:
                            for block in lora_blocks:
                                if f"blocks.{block}" in name:
                                    lora_params.append(name.replace(".weight", ""))
    
    lora_config = LoraConfig(
        r = args.lora_rank,
        lora_alpha = args.lora_alpha or args.lora_rank,
        init_lora_weights = "gaussian",
        target_modules = lora_params,
    )
    
    diffusion_model.add_adapter(lora_config, adapter_name="default")

    try:
        set_peft_model_state_dict(diffusion_model, lora_sd)
    except TypeError:  # older peft versions: activate the 'default' adapter first
        if hasattr(diffusion_model, "set_adapters"):
            diffusion_model.set_adapters("default")
        elif hasattr(diffusion_model, "set_active_adapters"):
            diffusion_model.set_active_adapters(["default"])
        set_peft_model_state_dict(diffusion_model, lora_sd)
        if hasattr(diffusion_model, "set_lora_scale"):
            diffusion_model.set_lora_scale(1.0)

    # sanity check: LoRA modules exist on the patch embedding and all LoRA keys match a module
    pe = diffusion_model.patch_embedding
    print("has lora_A:", hasattr(pe, "lora_A"), "has lora_B:", hasattr(pe, "lora_B"))
    name2mod = {n:m for n,m in diffusion_model.named_modules()}
    hit = miss = 0
    for k in lora_sd.keys():
        if not k.endswith(".weight"): 
            continue
        base = k[:-len(".weight")]
        for suf in (".lora_A", ".lora_B", ".lora_up", ".lora_down"):
            if base.endswith(suf):base = base[:-len(suf)]
        if base in name2mod:
            hit += 1
        else:
            miss += 1
    print(f"[preview] hit≈{hit}, miss≈{miss}")

    # ---- Repeated Residual Adaptation Head (RRAH) ----
    residual_head = None
    if args.residual_head_off:
        print("[load] residual head disabled (--residual_head_off)")
    else:
        if not (args.residual_head_path and os.path.exists(args.residual_head_path)):
            raise FileNotFoundError(f"residual head checkpoint not found: {args.residual_head_path} "
                                    "(pass --residual_head_off to run without RRAH)")
        residual_head = ResidualHead(
            in_channels=residual_in_channels(args.residual_input),
            base_model=diffusion_model,
            num_layers=args.residual_layers,
        ).to(device=device, dtype=torch.bfloat16)
        state = torch.load(args.residual_head_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "residual_head" in state:
            state = state["residual_head"]
        residual_head.load_state_dict(state)
        residual_head.eval()
        print(f"[load] residual head: {args.residual_head_path}")

    gc.collect()
    torch.cuda.empty_cache()
    
    torch.manual_seed(args.seed)
    
    
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(args.steps, device=device, shift=args.shift)

    # ===== batch (folder) or single video =====
    if args.control_video_dir:
        all_videos = [os.path.join(args.control_video_dir, f)
                      for f in sorted(os.listdir(args.control_video_dir))
                      if f.lower().endswith(".mp4")]
        if not all_videos:
            print(f"[WARN] no videos found in: {args.control_video_dir} ending with .mp4")
            return

        print(f"[INFO] found {len(all_videos)} videos ending with .mp4.")
        for vp in all_videos:
            try:
                save_path = generate_one(
                    control_path=vp,
                    real_output_dir=real_output_dir,
                    args=args,
                    device=device,
                    vae=vae,
                    diffusion_model=diffusion_model,
                    scheduler=scheduler,
                    positive_context=positive_context,
                    negative_context=negative_context,
                    residual_head=residual_head,
                )
                print(f"[OK] {vp} -> {save_path}")
            except Exception as e:
                print(f"[FAIL] {vp}: {e}")
                traceback.print_exc()

            # free memory between videos
            gc.collect()
            torch.cuda.empty_cache()
    else:
        if not args.control_video:
            raise ValueError("Either --control_video_dir or --control_video must be provided.")
        save_path = generate_one(
            control_path=args.control_video,
            real_output_dir=real_output_dir,
            args=args,
            device=device,
            vae=vae,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            positive_context=positive_context,
            negative_context=negative_context,
            residual_head=residual_head,
        )
        print(f"[OK] {args.control_video} -> {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description = "FADRA inference on LQ face videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B",
        help="Path to pretrained model or model identifier from huggingface",
    )
    parser.add_argument(
        "--lora",
        type = str,
        default = "./checkpoints/FADRA/wan-lora.safetensors",
        help = "LoRA checkpoint to load",
    )
    parser.add_argument(
        "--lora_scope",
        type=str,
        default="all",
        choices=["all", "patch_embedding"],
        help="Use all LoRA tensors or only patch_embedding LoRA tensors for ablation",
    )
    parser.add_argument(
        "--patch_embedding",
        type = str,
        default = "./checkpoints/FADRA/patch_embedding.pt",
        help = "Patch_embedding checkpoint to load",
    )

    parser.add_argument(
        "--output_dir",
        type = str,
        default = "./outputs/test",
        help = "Output directory for training results",
    )
    parser.add_argument(
        "--prompt",
        type = str,
        default = None,
        help = "Positive prompt to use instead of precalculated embedding",
    )
    parser.add_argument(
        "--negative_prompt",
        type = str,
        default = None,
        help = "Negative prompt to use instead of precalculated embedding",
    )
    parser.add_argument(
        "--embedding",
        type = str,
        default = "./embeddings/face-prompt_wan.safetensors",
        help = "Precalculated positive embedding file to use if no prompt is provided",
    )
    parser.add_argument(
        "--negative_embedding",
        type = str,
        default = "./embeddings/default_video_negative_wan.safetensors",
        help = "Precalculated negative embedding file to use if no negative prompt is provided",
    )
    parser.add_argument(
        "--seed",
        type = int,
        default = 42,
        help = "Seed for reproducible training",
    )
    parser.add_argument(
        "--cfg",
        type = float,
        default = 0.0,
        help = "CFG scale",
    )
    parser.add_argument(
        "--steps",
        type = int,
        default = 30,
        help = "Total number of inference steps",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default = 1.0,
        help = "Noise scheduler shift",
    )
    parser.add_argument(
        "--width",
        type = int,
        default = 624,
        help = "Generated width",
    )
    parser.add_argument(
        "--height",
        type = int,
        default = 624,
        help = "Generated height",
    )
    parser.add_argument(
        "--control_video",
        type = str,
        default = None,
        help = "Control signal video to use if lora is a control lora",
    )
    parser.add_argument(
        "--residual_head_path",
        type=str,
        default = "./checkpoints/FADRA/residual_head.pt",
        help="Residual head checkpoint to load",
    )
    parser.add_argument(
        "--residual_head_off",
        action="store_true",
        help="Disable residual head even if the checkpoint exists",
    )
    parser.add_argument(
        "--residual_input",
        type=str,
        default="pred+control",
        choices=["pred", "control", "pred+control"],
        help="Input for residual head: predicted v, control latents, or concatenation",
    )
    parser.add_argument(
        "--residual_layers",
        type=int,
        default=1,
        help="Number of DiT blocks in the residual head (RRAH)",
    )
    parser.add_argument(
        "--lora_targets",
        type = str,
        default = "self_attn, cross_attn",
        help = "layers to target with LoRA, default is self attention only",
    )
    parser.add_argument(
        "--lora_blocks",
        type = str,
        default = None,
        help = "blocks to target with LoRA, default is all blocks, otherwise comma separated string of block numbers",
    )
    parser.add_argument(
        "--lora_rank",
        type = int,
        default = 128,
        help = "The dimension of the LoRA update matrices",
    )
    parser.add_argument(
        "--lora_alpha",
        type = int,
        default = None,
        help = "The alpha value for LoRA, defaults to alpha=rank. Note: changing alpha will affect the learning rate, and if alpha=rank then changing rank will also affect learning rate",
    )
    parser.add_argument(
        "--control_video_dir",
        type=str,
        default=None,
        help="Process all videos in this directory (batch mode). If set, --control_video is ignored."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
