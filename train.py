import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import os
import gc
import math
import random
import numpy as np
import argparse
import json
import datetime
from tqdm import tqdm
from contextlib import contextmanager
from time import perf_counter
from glob import glob

from torchvision.transforms import v2
from safetensors.torch import load_file, save_file
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
import bitsandbytes as bnb

from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.modules.model import WanModel
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import cache_video

from utils.temp_rng import temp_rng
from utils.dataset import CombinedDataset
from utils.residual_head import ResidualHead, residual_in_channels
import wandb
import kornia.augmentation as K

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        use_ddp = True
    else:
        rank, world_size, local_rank, use_ddp = 0, 1, 0, False
    torch.backends.cuda.matmul.allow_tf32 = True
    return rank, world_size, local_rank, use_ddp

rank, world_size, local_rank, use_ddp = setup_distributed()
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
is_main = (rank == 0)

def make_dir(base, folder):
    new_dir = os.path.join(base, folder)
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


@contextmanager
def load_timer(target):
    print(f"loading {target}...")
    start_time = perf_counter()
    yield
    end_time = perf_counter()
    print(f"loaded {target} in {end_time - start_time:0.2f} seconds")


def download_model(args):
    from huggingface_hub import snapshot_download
    
    # get text encoder, they're all identical
    snapshot_download(
        repo_type = "model",
        repo_id = "Wan-AI/Wan2.1-I2V-14B-480P",
        local_dir = "./models/UMT5",
        allow_patterns = "google/*",
    )
    snapshot_download(
        repo_type = "model",
        repo_id = "Wan-AI/Wan2.1-I2V-14B-480P",
        local_dir = "./models/UMT5",
        allow_patterns = "models_t5_umt5-xxl-enc-bf16.pth",
        max_workers = 1,
    )
    
    # get vae, they're all identical
    snapshot_download(
        repo_type = "model",
        repo_id = "Wan-AI/Wan2.1-I2V-14B-480P",
        local_dir = "./models/vae",
        allow_patterns = "Wan2.1_VAE.pth",
        max_workers = 1,
    )
    
    # get clip vision if it's i2v
    if "-I2V-" in args.pretrained_model_name_or_path:
        snapshot_download(
            repo_type = "model",
            repo_id = "Wan-AI/Wan2.1-I2V-14B-480P",
            local_dir = "./models/clipvision",
            allow_patterns = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            max_workers = 1,
        )
        snapshot_download(
            repo_type = "model",
            repo_id = "Wan-AI/Wan2.1-I2V-14B-480P",
            local_dir = "./models/clipvision",
            allow_patterns = "xlm-roberta-large/*",
        )
    
    # get the correct diffusion model for HF identifier
    snapshot_download(
        repo_type = "model",
        repo_id = args.pretrained_model_name_or_path,
        local_dir = "./models/" + args.pretrained_model_name_or_path,
        allow_patterns = ["config.json", "diffusion_pytorch_model*"],
        max_workers = 1,
    )

@torch.inference_mode()
def cache_embeddings(args):
    if os.path.exists("./models/UMT5/models_t5_umt5-xxl-enc-bf16.pth"):
        with load_timer("UMT5 text encoder"):
            if os.path.exists("./models/UMT5/models_t5_umt5-xxl-enc-bf16.pth"):
                ckpt_dir = "./models/UMT5/models_t5_umt5-xxl-enc-bf16.pth"
                tk_dir = "./models/UMT5/google/umt5-xxl"
            else:
                ckpt_dir = args.pretrained_model_name_or_path
                tk_dir = args.pretrained_model_name_or_path + "/google/umt5-xxl"
            
            umt5_model = T5EncoderModel(
                text_len = 512,
                dtype = torch.bfloat16,
                device = device,
                checkpoint_path = ckpt_dir,
                tokenizer_path = tk_dir,
            )
    else:
        raise Exception("UMT5 model missing, download it first with --download_model")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # it would probably be better to dirwalk and batch the encoding
    # scanning all files at the start can be slow, and encoding one prompt at a time is inefficient
    caption_files = glob(os.path.join(args.dataset, "**", "*.txt" ), recursive=True)
    for file in tqdm(caption_files, desc="encoding captions") if is_main else caption_files:
        if not is_main: break
        embedding_path = os.path.splitext(file)[0] + "_wan.safetensors"
        
        if not os.path.exists(embedding_path):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    caption = f.read()
                    context = umt5_model([caption], umt5_model.device)[0]
                    embedding_dict = {"context": context}
                    save_file(embedding_dict, embedding_path)
            except UnicodeDecodeError as e:
                tqdm.write(f"unable to decode {file}: UnicodeDecodeError: {e}")
    
    del umt5_model
    gc.collect()
    torch.cuda.empty_cache()


def _load_latent_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pt":
        data = torch.load(path, map_location="cpu")
        latent = data.get("latent")
        meta = {k: v for k, v in data.items() if k != "latent"}
    elif ext == ".npy":
        latent = torch.from_numpy(np.load(path))
        meta_path = os.path.splitext(path)[0] + ".json"
        with open(meta_path, "r") as f:
            meta = json.load(f)
    else:
        raise ValueError(f"Unsupported latent format: {ext}")
    return latent, meta


def _resolve_embedding(video_path, dataset_root=None):
    if dataset_root and "train" in dataset_root:
        default_embedding = os.path.join(dataset_root, "face-prompt_wan.safetensors").replace("/train/", "/text-embedding/")
    elif dataset_root and "test" in dataset_root:
        default_embedding = os.path.join(dataset_root, "face-prompt_wan.safetensors").replace("/test/", "/text-embedding/")
    else:
        default_embedding = None

    candidate = os.path.splitext(video_path)[0] + "_wan.safetensors"
    if os.path.exists(candidate):
        embedding_file = candidate
    elif default_embedding and os.path.exists(default_embedding):
        embedding_file = default_embedding
    else:
        embedding_file = os.path.join(
            os.path.dirname(video_path),
            random.choice(["caption_original_wan.safetensors", "caption_florence_wan.safetensors"]),
        )
    if not os.path.exists(embedding_file):
        raise FileNotFoundError(f"No embedding file found for {video_path}")
    return embedding_file


class CachedLatentDataset(Dataset):
    def __init__(self, cache_dir, manifest_path=None, dataset_root=None):
        self.cache_dir = cache_dir
        self.dataset_root = dataset_root
        manifest_file = manifest_path or os.path.join(cache_dir, "manifest.json")
        with open(manifest_file, "r") as f:
            manifest = json.load(f)

        self.samples = []
        for video_key, entries in manifest.items():
            if entries:
                self.samples.append({
                    "video": video_key,
                    "entries": entries,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        chosen = random.choice(entry["entries"])
        hq_path = os.path.join(self.cache_dir, chosen["hq"])
        hq_latent, meta = _load_latent_file(hq_path)

        lq_name = random.choice(chosen["lq"])
        lq_path = os.path.join(self.cache_dir, lq_name)
        lq_latent, _ = _load_latent_file(lq_path)

        embedding_file = _resolve_embedding(meta["orig_video"], self.dataset_root)
        embedding_dict = load_file(embedding_file)

        return {
            "hq_latent": hq_latent,
            "lq_latent": lq_latent,
            "embedding_dict": embedding_dict,
        }




def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.resume_dir is not None:
        # continue training inside an existing run directory
        real_output_dir = args.resume_dir
        checkpoint_dir = make_dir(real_output_dir, "checkpoints")
        if is_main: print(f"[resume] using existing output dir: {real_output_dir}")
    else:
        # new run
        real_output_dir = make_dir(args.output_dir, f"{args.wandb_run_name}_{date_time}")
        checkpoint_dir = make_dir(real_output_dir, "checkpoints")
        if is_main: print(f"[new run] output dir: {real_output_dir}")

    t_writer = SummaryWriter(log_dir=real_output_dir, flush_secs=60)
    if args.wandb and is_main:
        import wandb
        run_name = f"{args.wandb_run_name}_{date_time}"
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        if args.resume_wandb_id:
            print(f"[resume] W&B run id: {args.resume_wandb_id}")
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                id=None, #id=args.resume_wandb_id,
                resume=False,#resume="allow",
                name=run_name,
                group=args.wandb_group,
                tags=tags,
                config=vars(args),
                dir=real_output_dir,
                save_code=True,
                reinit="finish_previous"
            )
        else:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                group=args.wandb_group,
                tags=tags,
                config=vars(args),
                dir=real_output_dir,
                save_code=True,
                reinit=True
            )
    if is_main:
        with open(os.path.join(real_output_dir, "command_args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=4)
    
    train_root = os.path.join(args.dataset, "train")
    if not os.path.exists(train_root):
        train_root = args.dataset
        if is_main:
            print(f"WARNING: train subfolder not found, using root folder {train_root} as train dataset")
    
    val_dataset = None
    for subfolder in ["val", "validation", "eval", "test"]:
        subfolder_path = os.path.join(args.dataset, subfolder)
        if os.path.exists(subfolder_path):
            val_dataset = subfolder_path
            break
    
    if val_dataset is None:
        val_dataset = args.dataset
        if is_main:
            print(f"WARNING: val/validation/test subfolder not found, using root folder {val_dataset} for stable loss validation")
            print("\033[33mThis will make it impossible to judge overfitting by the validation loss. Using a val split held out from training is highly recommended\033[m")
    
    with load_timer("train dataset"):
        if args.latent_cache_dir:
            train_dataset = CachedLatentDataset(
                cache_dir=args.latent_cache_dir,
                manifest_path=(os.path.join(args.latent_cache_dir, "manifest.json")
                               if os.path.exists(os.path.join(args.latent_cache_dir, "manifest.json"))
                               else os.path.join(args.dataset, "manifest.json")),
                dataset_root=train_root,
            )
        else:
            train_dataset = CombinedDataset(
                root_folder = train_root,
                token_limit = args.token_limit,
                max_frame_stride = args.max_frame_stride,
                random_frame_length = args.random_frame_length,
                bucket_resolutions = args.base_res,
            )
    with load_timer("validation dataset"):
        val_cache_dir = args.latent_cache_val_dir
        if val_cache_dir:
            val_dataset = CachedLatentDataset(
                cache_dir=val_cache_dir,
                manifest_path=os.path.join(val_cache_dir, "manifest.json"),
                dataset_root=val_dataset,
            )
        else:
            val_dataset = CombinedDataset(
                root_folder = val_dataset,
                token_limit = args.token_limit,
                limit_samples = args.val_samples,
                max_frame_stride = args.max_frame_stride,
                random_frame_length = args.random_frame_length,
                bucket_resolutions = args.base_res,
            )

    def collate_batch(batch):
        pixels = [sample["pixels"][0].movedim(0, 1) for sample in batch] # BFCHW -> FCHW -> CFHW
        context = [sample["embedding_dict"]["context"] for sample in batch]
        # LQ control is synthesized on the fly from the HQ pixels (see preprocess_control)
        return pixels, context, None
    
    def collate_cached(batch):
        hq_latents = [sample["hq_latent"] for sample in batch]
        context = [sample["embedding_dict"]["context"] for sample in batch]
        lq_latents = [sample["lq_latent"] for sample in batch]
        return hq_latents, context, lq_latents
    
    use_cached_train = isinstance(train_dataset, CachedLatentDataset)
    use_cached_val = isinstance(val_dataset, CachedLatentDataset)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                       shuffle=True, drop_last=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank,
                                     shuffle=False, drop_last=False) if use_ddp else None

    train_dataloader = DataLoader(train_dataset, shuffle=(train_sampler is None), sampler=train_sampler,
                                  collate_fn=collate_cached if use_cached_train else collate_batch,
                                  batch_size=args.train_batch_size, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False, sampler=val_sampler,
                                collate_fn=collate_cached if use_cached_val else collate_batch,
                                batch_size=args.test_batch_size, num_workers=8, pin_memory=True)
    
    with load_timer("VAE"):
        if os.path.exists("./models/vae/Wan2.1_VAE.pth"):
            ckpt_dir = "./models/vae/Wan2.1_VAE.pth"
        else:
            ckpt_dir = args.pretrained_model_name_or_path
        
        vae = WanVAE(vae_pth=ckpt_dir, dtype=torch.bfloat16, device=device)
        
        gc.collect()
        torch.cuda.empty_cache()
    
    with load_timer("diffusion model"):
        if os.path.exists("./models/" + args.pretrained_model_name_or_path):
            ckpt_dir = "./models/" + args.pretrained_model_name_or_path
        else:
            ckpt_dir = args.pretrained_model_name_or_path
        
        diffusion_model = WanModel.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16).to(device)
        diffusion_model.requires_grad_(False)
        
        if args.gradient_checkpointing:
            diffusion_model.enable_gradient_checkpointing()
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # LQ pixel-alignment feature fusion: expand the input 3D patch embedding from 16 to 32
    # channels (noisy latent + LQ latent); the new LQ channels are zero-initialized.
    if args.control_lora:
        with torch.no_grad():
            in_cls = diffusion_model.patch_embedding.__class__ # nn.Conv3d
            old_in_dim = diffusion_model.in_dim # 16
            new_in_dim = old_in_dim * 2
            
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
    if args.resume_dir is not None:
        # when resuming, optionally restore the trained 32-channel patch embedding
        if args.pe_ckpt_path and os.path.exists(args.pe_ckpt_path):
            if is_main:
                print(f"[resume] loading patch embedding: {args.pe_ckpt_path}")
            pe_dict = torch.load(args.pe_ckpt_path, map_location="cpu")
            spec = pe_dict.get("spec", None)
            if spec is not None:
                assert tuple(diffusion_model.patch_embedding.kernel_size) == tuple(spec["kernel_size"])
                assert tuple(diffusion_model.patch_embedding.stride)      == tuple(spec["stride"])
                assert tuple(diffusion_model.patch_embedding.padding)     == tuple(spec["padding"])
            with torch.no_grad():
                diffusion_model.patch_embedding.weight.copy_(
                    pe_dict["weight"].to(diffusion_model.patch_embedding.weight.device,
                                     diffusion_model.patch_embedding.weight.dtype)
                )
                if pe_dict.get("bias") is not None and diffusion_model.patch_embedding.bias is not None:
                    diffusion_model.patch_embedding.bias.copy_(
                        pe_dict["bias"].to(diffusion_model.patch_embedding.bias.device,
                                       diffusion_model.patch_embedding.bias.dtype)
                  )
        else:
            if is_main:
                if args.pe_ckpt_path:
                    print(f"[resume] patch_embedding ckpt not found: {args.pe_ckpt_path} (using init)")
                else:
                    print("[resume] no patch_embedding ckpt provided (using init)")

    lora_params = []
    if args.control_lora:
        lora_params.append("patch_embedding")
    
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

    def load_last_bundle(checkpoint_dir, diffusion_model, residual_head,
                     optimizer_dict, train_parameters,
                     lr_scheduler=None, scaler=None):
        """
        Restore a full training state from checkpoints/latest_training_state.pt:
        - LoRA weights (from state['lora'])
        - the per-parameter optimizers (optimizer_dict)
        - optional lr_scheduler / AMP scaler
        - RNG states and global step
        Returns the restored global_step (0 if nothing was found).
        """
        last_pt = os.path.join(checkpoint_dir, "latest_training_state.pt")
        if not os.path.exists(last_pt):
            if is_main: print(f"[resume] last bundle not found: {last_pt}")
            return 0

        if is_main: print(f"[resume] loading LAST bundle: {last_pt}")
        state = torch.load(last_pt, map_location="cpu", weights_only=False)

        # 1) LoRA weights
        lora_sd = state.get("lora", None)
        if lora_sd is None:
            if is_main: print("[resume] last.pt has no 'lora' field; skip")
        else:
            outcome = set_peft_model_state_dict(diffusion_model, lora_sd)
            if is_main and len(outcome.unexpected_keys) > 0:
                print("[resume] warn: some LoRA keys unmatched:")
                for k in outcome.unexpected_keys:
                    print("  -", k)

        # 2) residual head weights
        if residual_head is not None and state.get("residual_head") is not None:
            try:
                residual_head.load_state_dict(state["residual_head"])
            except Exception as e:
                if is_main: print(f"[resume] residual_head.load_state_dict failed: {e}")

        # 3) optimizers, matched by the order of train_parameters
        opt_pack = state.get("optimizer_dict", {})
        restored = 0
        for idx, (p, _) in enumerate(train_parameters):
            key = str(idx)
            if key in opt_pack:
                try:
                    optimizer_dict[p].load_state_dict(opt_pack[key])
                    restored += 1
                except Exception as e:
                    if is_main: print(f"[resume] optimizer[{idx}].load_state_dict failed: {e}")
        if is_main: print(f"[resume] optimizers restored: {restored}/{len(train_parameters)}")

        # 4) lr_scheduler (not used currently)
        if lr_scheduler is not None and state.get("lr_scheduler") is not None:
            try:
                lr_scheduler.load_state_dict(state["lr_scheduler"])
            except Exception as e:
                if is_main: print(f"[resume] lr_scheduler.load_state_dict failed: {e}")

        # 5) AMP scaler (not used currently)
        if scaler is not None and ("scaler" in state):
            try:
                scaler.load_state_dict(state["scaler"])
            except Exception as e:
                if is_main: print(f"[resume] scaler.load_state_dict failed: {e}")

        # 6) RNG & step
        try:
            random.setstate(state["rng_python"])
            np.random.set_state(state["rng_numpy"])
            torch.set_rng_state(state["rng_torch"])
            if torch.cuda.is_available() and state.get("rng_cuda") is not None:
                torch.cuda.set_rng_state_all(state["rng_cuda"])
        except Exception as e:
            if is_main: print(f"[resume] RNG restore failed: {e}")

        gs = int(state.get("global_step", 0))
        if is_main: print(f"[resume] global_step (from last): {gs}")
        return gs
    

    # ---- DDP wrap ----
    model_no_ddp = diffusion_model
    if use_ddp:
        ddp_model = DDP(model_no_ddp, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    else:
        ddp_model = model_no_ddp

    def M_ddp():
        return ddp_model.module if use_ddp else ddp_model

    def F_ddp(*args, **kwargs):
        return ddp_model(*args, **kwargs)

    # ---- Repeated Residual Adaptation Head (RRAH) ----
    residual_head = None
    if args.residual_head:
        residual_head = ResidualHead(
            in_channels=residual_in_channels(args.residual_input),
            base_model=diffusion_model,
            num_layers=args.residual_layers,
        ).to(device=device, dtype=torch.bfloat16)
        if use_ddp:
            residual_head = DDP(residual_head, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    def R_ddp():
        return residual_head.module if (use_ddp and residual_head is not None) else residual_head

    total_parameters = 0
    train_parameters = []
    
    for name, param in diffusion_model.named_parameters():
        if param.requires_grad:
            lr = args.learning_rate
            if "patch_embedding" in name:
                lr *= args.input_lr_scale
            param.data = param.to(torch.float32)
            train_parameters.append((param, lr))
            total_parameters += param.numel()

    if args.residual_head and residual_head is not None:
        residual_lr = args.residual_lr if args.residual_lr is not None else args.learning_rate
        for name, param in R_ddp().named_parameters():
            if param.requires_grad:
                # keep residual head dtype (bf16 if you set it) to avoid dtype mismatch
                param.data = param.to(param.dtype)
                train_parameters.append((param, residual_lr))
                total_parameters += param.numel()
    
    if is_main: print(f"total trainable parameters: {total_parameters:,}")
    if is_main and args.wandb:
        wandb.summary["trainable_params"] = total_parameters
        wandb.config.update({
            "lora_targets_resolved": lora_params,
            "control_lora": args.control_lora,
        }, allow_val_change=True)
        wandb.watch(M_ddp(), log="gradients", log_freq=100, log_graph=False)

    # Instead of having just one optimizer, we will have a dict of optimizers
    # for every parameter so we could reference them in our hook.
    optimizer_dict = {p: bnb.optim.AdamW8bit([p], lr=lr) for p, lr in train_parameters}
    
    # Define our hook, which will call the optimizer step() and zero_grad()
    def optimizer_hook(parameter) -> None:
        optimizer_dict[parameter].step()
        optimizer_dict[parameter].zero_grad()
    
    # Register the hook onto every trainable parameter
    for p, _ in train_parameters:
        p.register_post_accumulate_grad_hook(optimizer_hook)

    # ----- resume from latest_training_state.pt -----
    global_step = 0
    if args.resume_dir is not None:
        # if latest_training_state.pt exists, restore full state
        last_path = os.path.join(checkpoint_dir, "latest_training_state.pt")
        if is_main: print(f"[resume] checking for latest bundle: {last_path}")
        if os.path.exists(last_path):
            gs_last = load_last_bundle(
                checkpoint_dir=checkpoint_dir,
                diffusion_model=diffusion_model,
                residual_head=R_ddp() if args.residual_head else None,
                optimizer_dict=optimizer_dict if 'optimizer_dict' in locals() else {},
                train_parameters=train_parameters if 'train_parameters' in locals() else [],
                lr_scheduler=None,
                scaler=None
            )
            try:
                global_step = max(int(global_step), int(gs_last))
            except Exception:
                global_step = int(gs_last)
            if is_main:  print(f"[resume] global_step now = {global_step}")
    
    context_negative = load_file(args.distill_negative)["context"].to(dtype=torch.bfloat16, device=device)
    
    # --- LQ synthesis: bilinear down/up-sampling + Gaussian blur, Gaussian noise, JPEG ---
    # One set of degradation parameters is sampled per clip and shared by all frames.
    def preprocess_control(control):
        # make sure the input is (B, F, C, H, W)
        if control.shape[0] == 3 and control.shape[1] != 3:
            control = control.movedim(0, 1).unsqueeze(0)  # CFHW -> BFCHW
        height, width = control.shape[-2:]
        kernel_size_ = random.choice(range(3, 27, 2))  # odd kernel size in [3, 25]
        blur_strength = random.uniform(1.5,6)
        resize_ratio   = random.uniform(1.0,8.0)
        down_h = max(1, int(round(height / resize_ratio)))
        down_w = max(1, int(round(width / resize_ratio)))
        noise_sigma    = random.uniform(0.01,0.05)
        random.uniform(20, 35)  # unused (former CRF value); kept so the RNG stream matches the paper code

        degrade = v2.Compose([
            v2.Resize(size=(down_h, down_w)),
            v2.Resize(size=(height, width)),
            v2.GaussianBlur(kernel_size=kernel_size_, sigma=blur_strength),
        ])

        degraded = degrade(control)
        degraded = torch.clamp(torch.nan_to_num(degraded), -1, 1)

        x01 = (degraded * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
        orig_dtype = x01.dtype
        x01 = x01.to(torch.float32)               # kornia does not support bfloat16
        x01 = v2.GaussianNoise(mean=0.0, sigma=noise_sigma, clip=True)(x01)
        # JPEG compression on GPU
        jpeg = K.RandomJPEG(jpeg_quality=(25.0, 85.0), p=1.0, keepdim=True).to(x01.device)
        y01 = jpeg(x01[0])  # (F,3,H,W) float in [0,1]
        degraded = (y01 * 2.0 - 1.0).clamp(-1, 1).to(orig_dtype)  # [0,1] -> [-1,1]

        degraded = torch.clamp(torch.nan_to_num(degraded), -1, 1)

        degraded = degraded.movedim(0,1)   # (F,C,H,W) -> (C,F,H,W)
        return degraded
    
    def prepare_conditions(batch, timestep=None, use_cached=False):
        data, context, control = batch
        
        context = [c.to(dtype=torch.bfloat16, device=device) for c in context]
        
        timings = {}
        if use_cached:
            latents = [p.to(dtype=torch.bfloat16, device=device) for p in data]
            noise = [torch.randn_like(l) for l in latents]
        else:
            pixels  = [p.to(dtype=torch.bfloat16, device=device) for p in data]
            start_vae = perf_counter()
            latents = vae.encode(pixels)
            timings["vae_encode"] = perf_counter() - start_vae
            noise = [torch.randn_like(l) for l in latents]
        
        if timestep is not None:
            timesteps = torch.ones(len(latents)).to(device) * timestep
            sigmas = timesteps.float() / 1000
        else:
            sigmas = torch.rand(len(latents)).to(device)
            sigmas = (args.shift * sigmas) / (1 + (args.shift - 1) * sigmas)
            timesteps = torch.round(sigmas * 1000).long()
            sigmas = timesteps.float() / 1000
        
        control_latents = None
        if args.control_lora:
            if use_cached:
                # cached LQ latents
                control_latents = [p.to(dtype=torch.bfloat16, device=device) for p in control] if control is not None else None
            else:
                # synthesize LQ videos from the HQ pixels, then encode them
                control = [preprocess_control(p) for p in pixels]
                start_control_vae = perf_counter()
                control_latents = vae.encode(control)
                timings["control_vae_encode"] = perf_counter() - start_control_vae

        target_vel = []
        target_dwt = []
        noisy_model_input = []
        for i in range(len(latents)):
            # flow-matching velocity target: noise - clean latent
            target_vel.append(noise[i] - latents[i])
            # clean latent z (target of the frequency-aware loss; key name kept from the original code)
            target_dwt.append(latents[i])

            noisy = noise[i] * sigmas[i] + latents[i] * (1 - sigmas[i])
            
            if args.control_lora:
                noisy = torch.cat([noisy, control_latents[i]], dim=0) # CFHW, so channel dim is 0
            
            noisy_model_input.append(noisy.to(torch.bfloat16))
        
        return {
            "sigmas": sigmas,
            "target_vel": target_vel,
            "target_dwt": target_dwt,
            "context": context,
            "timesteps": timesteps,
            "noisy_model_input": noisy_model_input,
            "control_latents": control_latents,
            "timings": timings,
        }
    
    def freq_loss(pred: torch.Tensor,
              target: torch.Tensor,
              block_size: int = 8,
              quality: int = 85,
              mode: str = 'inv_gamma',
              gamma: float = 1.0):
        """Frequency-Aware Loss (FAL): per-frame 8x8 block-DCT, weighted squared L2.

        The weights come from the JPEG luminance quantization table (quality / gamma below).
        For Wan latents ([B, 16, F, H, W]) no colour transform is applied.
        """
        if pred.dim() == 5:
            # [B,T,C,H,W] or [B,C,F,H,W]
            channels_guess = pred.shape[2] if pred.shape[1] == 6 else pred.shape[1]
        elif pred.dim() == 4:
            channels_guess = pred.shape[1]
        else:
            raise ValueError(f"Unsupported input dim {pred.dim()}, expected 4 or 5")

        if channels_guess == 3:
            target_ycbcr = _rgb2ycbcr(target)
            pred_ycbcr   = _rgb2ycbcr(pred)
        else:
            target_ycbcr = target
            pred_ycbcr   = pred

        v_t_freq = _dct(target_ycbcr, block_size=block_size)
        out_freq = _dct(pred_ycbcr,   block_size=block_size)

        # Frequency weights. For latents out_freq is [B, C, F, Bh, Bw, bs, bs] and shape[2] is the
        # latent-frame axis; the luminance weights are identical along that axis, so every
        # channel / frame receives the same JPEG-luminance-derived weighting.
        if out_freq.shape[2] == 3:
            w = _build_freq_weight(
                quality=quality,
                mode=mode,
                gamma=gamma,
                device=out_freq.device,
                dtype=out_freq.dtype,
            )
        else:
            # replicate the luminance weights
            C = out_freq.shape[2]
            w_base = _build_freq_weight(
                quality=quality,
                mode=mode,
                gamma=gamma,
                device=out_freq.device,
                dtype=out_freq.dtype,
            )  # (1,3,1,1,bs,bs)
            w_y = w_base[:, :1]  # luminance (1,1,1,1,bs,bs)
            w = w_y.expand(1, C, 1, 1, block_size, block_size)  # (1,C,1,1,bs,bs)

        fm_loss_freq = (w * (out_freq - v_t_freq).pow(2)).mean()
        return fm_loss_freq

    # ===== DCT / freq utils =====
    def _create_dct_matrix(N: int, device=None, dtype=torch.float32):
        """
        Orthonormal NxN DCT-II matrix.
        """
        n = torch.arange(N, dtype=dtype, device=device)
        k = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)  # [N,1]
        C = torch.cos(math.pi * (2 * n + 1) * k / (2.0 * N))          # [N,N]

        alpha = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device) / N) * torch.ones(N, dtype=dtype, device=device)
        alpha[0] = math.sqrt(1.0 / N)

        C = alpha.unsqueeze(1) * C  # [N,N]
        return C
    
    def _rgb2ycbcr(x: torch.Tensor):
        """
        RGB -> YCbCr

        Supports images [B, C, H, W] and videos [B, T, C, H, W] / [B, C, F, H, W].
        Inputs that are not 3-channel (e.g. 16-channel latents) are passed through unchanged.
        """
        if x.dim() == 5:
            B0, D1, D2, H, W = x.shape

            if D2 == 16:
                # [B, T, C, H, W]
                B, T, C, H, W = x.shape
                x = x.view(B * T, C, H, W)
                layout = 'BTC'   # B,T,C
                BTC_num = (B, T, C)

            elif D1 == 16:
                # [B, C, F, H, W] -> [B, F, C, H, W]
                B, C, F, H, W = x.shape
                x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W]
                B, T, C, H, W = x.shape                    # T = F
                x = x.view(B * T, C, H, W)                 # [B*T, C, H, W]
                layout = 'BCF'
                BTC_num = (B, T, C)
            else:
                raise ValueError(
                    f"Unsupported 5D layout {x.shape}, "
                    "expect [B,T,C,H,W] or [B,C,F,H,W] with C==16"
                )

            is_video = True

        elif x.dim() == 4:
            # image: [B, C, H, W]
            B, C, H, W = x.shape
            is_video = False
        else:
            raise ValueError(f"Unsupported input dim {x.dim()}, expected 4 or 5")

        if x.shape[1] == 3:
            # standard RGB -> YCbCr
            r = x[:, 0:1, :, :]
            g = x[:, 1:2, :, :]
            b = x[:, 2:3, :, :]

            y  = 0.299 * r + 0.587 * g + 0.114 * b
            cb = -0.168736 * r - 0.331264 * g + 0.5 * b
            cr = 0.5 * r - 0.418688 * g - 0.081312 * b

            out = torch.cat([y, cb, cr], dim=1)  # [B', 3, H, W]
        else:
            # non-RGB input (e.g. 16-channel latent): pass through
            out = x

        if is_video:
            B, T, C = BTC_num
            if layout == 'BTC':
                out = out.view(B, T, out.shape[1], H, W)
            elif layout == 'BCF':
                out = out.view(B, T, out.shape[1], H, W).permute(0, 2, 1, 3, 4).contiguous()
            else:
                raise RuntimeError("Unknown layout flag")
        
        return out
    
    
    def _dct(x: torch.Tensor, block_size: int = 8):
        """
        8x8 block DCT

        image: x [B, C, H, W]    -> [B, C, Bh, Bw, bs, bs]
        video: x [B, C, F, H, W] -> [B, C, F, Bh, Bw, bs, bs]
        """
        bs = block_size
        layout = None

        # ---- flatten the time axis ----
        if x.dim() == 5:
            # video: [B, T, C, H, W]
            B0, D1, D2, H, W = x.shape
            if D2 == 16:
                # [B, T, C, H, W]
                B, T, C, H, W = x.shape
                # assert C == 16, f"channel dim must be 16, got {C}"
                x = x.view(B * T, C, H, W)    # [B*T, C, H, W]
                is_video = True
                layout = 'BTC'
            elif D1 == 16:
                # [B, C, F, H, W]
                B, C, num_frames, H, W = x.shape
                # assert C == 16
                x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W]
                B, T, C, H, W = x.shape                    # T = F
                x = x.view(B * T, C, H, W)                 # [B*T, C, H, W]
                is_video = True
                layout = 'BCF'
                BTC_num = (B, T, C)
            else:
                raise ValueError(
                    f"Unsupported 5D layout {x.shape}, "
                    "expect [B,T,C,H,W] or [B,C,F,H,W]"
                )
    
        elif x.dim() == 4:
            # image: [B, C, H, W]
            B, C, H, W = x.shape
            is_video = False
        else:
            raise ValueError(f"Unsupported input dim {x.dim()}, expected 4 or 5")

        # ---- per-frame block DCT ----
        pad_h = (-H) % bs
        pad_w = (-W) % bs
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        B_flat, C, H2, W2 = x.shape
        Bh, Bw = H2 // bs, W2 // bs

        # (B_flat, C, Bh, Bw, bs, bs)
        blocks = x.unfold(2, bs, bs).unfold(3, bs, bs)
        blocks = blocks.contiguous().view(-1, bs, bs)  # [B_flat*C*Bh*Bw, bs, bs]

        # ---- DCT basis ----
        Cmat = _create_dct_matrix(bs, device=x.device, dtype=x.dtype)  # [bs, bs]

        dct_flat = torch.matmul(Cmat.unsqueeze(0), blocks)
        dct_flat = torch.matmul(dct_flat, Cmat.t().unsqueeze(0))

        dct = dct_flat.view(B_flat, C, Bh, Bw, bs, bs)  # [B_flat, C, Bh, Bw, bs, bs]

        # ---- restore the time axis ----
        if is_video:
            B, T, C = BTC_num
            # [B, T, C, Bh, Bw, bs, bs]
            dct = dct.view(B, T, C, Bh, Bw, bs, bs)

            if layout == 'BTC':
                pass
            elif layout == 'BCF':
                # [B, T(=F), C, ...] -> [B, C, F, ...]
                dct = dct.permute(0, 2, 1, 3, 4, 5, 6).contiguous()
            else:
                raise RuntimeError("Unknown layout flag")
        
        return dct

    def _build_freq_weight(quality=85, mode='inv_gamma', gamma=1.0,device=None, dtype=torch.float32):
        # JPEG luminance & chrominance base tables
        lum_q = torch.tensor([
            [16,11,10,16,24,40,51,61],
            [12,12,14,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],
            [14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],
            [24,35,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],
            [72,92,95,98,112,100,103,99]], dtype=torch.float32, device=device)

        chr_q = torch.tensor([
            [17,18,24,47,99,99,99,99],
            [18,21,26,66,99,99,99,99],
            [24,26,56,99,99,99,99,99],
            [47,66,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99]], dtype=torch.float32, device=device)

        def scale_q(base_q, quality):
            q = max(1, min(100, int(quality)))
            if q < 50:
                scale = 5000 / q
            else:
                scale = 200 - 2*q
            return torch.floor((base_q*scale + 50)/100).clamp(1,255)

        Q_y = scale_q(lum_q, quality)
        Q_cbcr = scale_q(chr_q, quality)

        def q_to_weight(Q):
            if mode == 'inv':
                w = 1.0 / Q
            elif mode == 'inv_gamma':
                w = (Q.mean()/Q) ** gamma
            else:
                raise ValueError("mode must be 'inv' or 'inv_gamma'")
            return w / w.mean()

        # Y -> luminance, Cb/Cr -> chrominance
        w_y = q_to_weight(Q_y)
        w_cb = q_to_weight(Q_cbcr)
        w_cr = q_to_weight(Q_cbcr)

        # stack to (1,3,1,1,8,8)
        w = torch.stack([w_y, w_cb, w_cr], dim=0)
        w = w.to(dtype=dtype)
        return w.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # (1,3,1,1,8,8)

    
    def predict_loss(conditions):
        target_vel = torch.stack(conditions["target_vel"])
        target_frq_on_x = torch.stack(conditions["target_dwt"]) if args.use_freq_loss else None
        sigmas = conditions["sigmas"]
        noisy_model_input = conditions["noisy_model_input"]
        c, f, h, w = noisy_model_input[0].shape
        seq_len = math.ceil((h / 2) * (w / 2) * f)
        timings = conditions.setdefault("timings", {})
        
        start_diffusion = perf_counter()
        pred = torch.stack(
            F_ddp(
                x = noisy_model_input,
                t = conditions["timesteps"],
                context = conditions["context"],
                seq_len = seq_len,
            )
        )
        timings["diffusion_forward"] = perf_counter() - start_diffusion

        # RRAH: predict a residual correction of the velocity
        delta = None
        if args.residual_head and residual_head is not None:
            control_latents = conditions.get("control_latents")
            if args.residual_input == "pred":
                res_in = pred
            elif args.residual_input == "control":
                if control_latents is None:
                    res_in = torch.zeros_like(pred)
                else:
                    res_in = torch.stack(control_latents).to(dtype=pred.dtype, device=pred.device)
            elif args.residual_input == "pred+control":
                if control_latents is None:
                    control_stack = torch.zeros_like(pred)
                else:
                    control_stack = torch.stack(control_latents).to(dtype=pred.dtype, device=pred.device)
                res_in = torch.cat([pred, control_stack], dim=1)
            else:
                raise ValueError(f"Unknown residual_input: {args.residual_input}")

            delta = residual_head(
                res_in.to(dtype=torch.bfloat16, device=device),
                t=conditions["timesteps"],
                context=conditions["context"],
                seq_len=seq_len,
            ).to(dtype=pred.dtype)
            pred = pred + delta
        
        assert not torch.isnan(pred).any()

        loss_mse = F.mse_loss(pred.float(), target_vel.float())
        loss_final = loss_mse
        if args.use_freq_loss:
            latent_c = pred.shape[1]
            # FAL on z: recover the clean-latent estimate z = x_t - sigma * v and compare with the clean latent
            pred_latent = torch.stack([p * (-sigmas[i]) + noisy_model_input[i][:latent_c] for i, p in enumerate(pred)])
            target_latent = torch.stack([t for t in target_frq_on_x])
            loss_freq = freq_loss(pred_latent.float(), target_latent.float())
            loss_freq_w = args.freq_loss_weight * loss_freq
            if is_main and args.wandb:
                wandb.log({"loss/freq_weighted": float(loss_freq_w.item())}, step=global_step)
            if is_main and t_writer:
                t_writer.add_scalar("loss/freq_weighted", float(loss_freq_w.item()), global_step)
            loss_final = loss_final + loss_freq_w

        return loss_final
    
    # --- visualize a single validation batch: side-by-side control | decoded ---
    @torch.inference_mode()
    def visualize_val_batch(batch, out_file, infer_steps=None, fps=16):
        # batch: (pixels(list[CFHW]), context(list[tensor]), control(list[CFHW] or None))
        pixels, context_pos, control = batch
        pixels  = [p.to(dtype=torch.bfloat16, device=device) for p in pixels]
        context_pos = [c.to(dtype=torch.bfloat16, device=device) for c in context_pos]
        # negative context (the embedding passed via --distill_negative)
        context_neg = [context_negative] * len(pixels)
        # visualize the first sample only
        p = pixels[0]            # CFHW
        C, F, H, W = p.shape

        # Wan latent size (same as the inference scripts)
        latent_width  = (W // 16) * 2
        latent_height = (H // 16) * 2
        latent_frames = (F - 1) // 4 + 1
        width  = latent_width  * 8
        height = latent_height * 8
        frames = (latent_frames - 1) * 4 + 1
        seq_len = math.ceil((latent_height / 2) * (latent_width / 2) * latent_frames)

        # synthesize the LQ input with the training degradation
        if args.control_lora:
            if control is not None and control[0] is not None:
                cpx = control[0].to(dtype=torch.bfloat16, device=device)
            else:
                cpx = preprocess_control(p)  # CFHW
        else:
            cpx = None
        # sampler
        steps = infer_steps or args.val_vis_infer_steps
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(steps, device=device, shift=args.shift)

        # initial noise latent
        torch.manual_seed(args.val_seed or args.seed)
        latents = torch.randn(16, latent_frames, latent_height, latent_width, device=device)

        if cpx is not None:
            control_latent = vae.encode([cpx])[0].to(device)
            assert control_latent.shape == latents.shape, f"{control_latent.shape} {latents.shape}"

        # positive / negative context (first sample of the batch)
        pos_ctx = [context_pos[0]]
        neg_ctx = [context_neg[0]]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for t in scheduler.timesteps:
                # model input: [noisy latent, LQ latent] along the channel dim
                noisy_in = [torch.cat([latents, control_latent], dim=0)] if args.control_lora else [latents]
                timestep = torch.stack([t])

                # cond / uncond passes (cfg scale 0 -> only the uncond branch is used, as at inference)
                noise_pred_cond = F_ddp(
                    x = noisy_in, t = timestep, context = pos_ctx, seq_len = seq_len
                )[0]
                noise_pred_uncond = F_ddp(
                    x = noisy_in, t = timestep, context = neg_ctx, seq_len = seq_len
                )[0]
                
                delta = None
                if args.residual_head and residual_head is not None:
                    control_latents = control_latent
                if args.residual_input == "pred":
                    res_in = noise_pred_uncond
                elif args.residual_input == "control":
                    if control_latents is None:
                        res_in = torch.zeros_like(noise_pred_uncond)
                    else:
                        res_in = torch.stack(control_latents).to(dtype=noise_pred_uncond.dtype, device=noise_pred_uncond.device)
                elif args.residual_input == "pred+control":
                    if control_latents is None:
                        control_stack = torch.zeros_like(noise_pred_uncond)
                    else:
                        control_stack = (control_latents).to(dtype=noise_pred_uncond.dtype, device=noise_pred_uncond.device) #torch.stack(control_latents).to(dtype=noise_pred_uncond.dtype, device=noise_pred_uncond.device)
                        res_in = torch.cat([noise_pred_uncond, control_stack], dim=0)
                else:
                    raise ValueError(f"Unknown residual_input: {args.residual_input}")

                delta = residual_head(
                    res_in.to(dtype=torch.bfloat16, device=device),
                    t=timestep,
                    context=neg_ctx,
                    seq_len=seq_len,
                ).to(dtype=noise_pred_uncond.dtype)
                noise_pred_uncond = noise_pred_uncond + delta

                noise_pred = noise_pred_uncond + 0.0 * (noise_pred_cond - noise_pred_uncond)

                latents = scheduler.step(
                    noise_pred.unsqueeze(0),
                    timestep,
                    latents.unsqueeze(0),
                    return_dict=False,
                )[0].squeeze(0)
            decoded = vae.decode([latents])[0].to(device)      # CFHW in [-1, 1]
            if cpx is None:
                vis = decoded
            else:
                # HQ | LQ | output, concatenated along the longer side
                cat_dim = -2 if W > H else -1
                vis = torch.cat([p.to(device),cpx.to(device), decoded], dim=cat_dim)
            cache_video(
                tensor = vis[None],    # B=1, CFHW
                save_file = out_file,
                fps = fps,
                nrow = 1,
                normalize = True,
                value_range = (-1, 1),
            )
    


    def save_last_bundle(checkpoint_dir, global_step, diffusion_model, residual_head,
                     optimizer_dict, lr_scheduler=None, scaler=None):
        """
        Save the "latest" weights (LoRA / patch embedding / RRAH) for inference, plus a
        full training state (optimizers, RNG, global step) for resuming.
        """
        pe_path = os.path.join(checkpoint_dir, "patch_embedding-32-latest.pt")
        patch_embedding = {
            "in_dim": getattr(diffusion_model, "in_dim", None),
            "weight": diffusion_model.patch_embedding.weight.detach().cpu(),
            "bias":   (diffusion_model.patch_embedding.bias.detach().cpu()
                       if diffusion_model.patch_embedding.bias is not None else None),
        }
        torch.save(patch_embedding, pe_path)

        # 1) LoRA weights
        lora_sd = get_peft_model_state_dict(diffusion_model)
        last_lora = os.path.join(checkpoint_dir, "wan-lora-latest.safetensors")
        save_file(lora_sd, last_lora)

        if residual_head is not None:
            torch.save(residual_head.state_dict(), os.path.join(checkpoint_dir, f"residual_head-latest.pt"))

        # 2) per-parameter optimizers, stored by index (the trainable-parameter order must match on resume)
        opt_state_packed = {}
        for idx, (p, _) in enumerate(train_parameters):
            opt = optimizer_dict[p]
            opt_state_packed[str(idx)] = opt.state_dict()

        # 3) training state / RNG / optional scaler
        last_bundle = {
            "global_step": int(global_step),
            "lora": lora_sd,
            "patch_embedding": patch_embedding,
            "residual_head": (residual_head.state_dict() if residual_head is not None else None),
            "optimizer_dict": opt_state_packed,
            "lr_scheduler": (lr_scheduler.state_dict() if lr_scheduler is not None else None),
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        }
        if scaler is not None:
            try:
                last_bundle["scaler"] = scaler.state_dict()
            except Exception:
                pass

        latest_pt = os.path.join(checkpoint_dir, "latest_training_state.pt")
        torch.save(last_bundle, latest_pt)
        if is_main: print(f"[latest] saved: {latest_pt}")


    gc.collect()
    torch.cuda.empty_cache()
    ddp_model.train()
    if args.residual_head and residual_head is not None:
        residual_head.train()
    
    progress_bar = tqdm(total=args.max_train_steps, initial=global_step)
    while global_step < args.max_train_steps:
        for step, batch in enumerate(train_dataloader):
            start_step = perf_counter()
            with torch.inference_mode():
                conditions = prepare_conditions(batch, use_cached=use_cached_train)
            
            gc.collect()
            torch.cuda.empty_cache()
            
            loss = predict_loss(conditions)
            if is_main and t_writer:
                t_writer.add_scalar("loss/train", loss.detach().item(), global_step)
            if is_main and args.wandb:
                wandb.log({
                    "loss/train": float(loss.detach().item()),
                }, step=global_step)
            loss.backward()  # parameters are updated inside the per-parameter optimizer hooks
            
            gc.collect()
            torch.cuda.empty_cache()
            
            if is_main and t_writer:
                t_writer.add_scalar("debug/step_time", perf_counter() - start_step, global_step)
            if is_main and args.wandb:
                wandb.log({
                    "debug/step_time": float(perf_counter() - start_step)
                }, step=global_step)
            if is_main and progress_bar:
                progress_bar.update(1)
            global_step += 1
            
            if global_step == 1 or global_step % args.val_steps == 0:
                if is_main:
                    with torch.inference_mode(), temp_rng(args.val_seed or args.seed):
                        val_loss = 0.0
                        for step, batch in enumerate(tqdm(val_dataloader, desc="validation", leave=False)):
                            conditions = prepare_conditions(batch, timestep=args.val_timestep, use_cached=use_cached_val)
                            loss = predict_loss(conditions)
                            val_loss += loss.detach().item()
                    t_writer.add_scalar("loss/validation", val_loss / len(val_dataloader), global_step)
                    if args.wandb:
                        wandb.log({
                            "loss/validation": float(val_loss / len(val_dataloader))
                        }, step=global_step)
                    if args.val_vis:
                        vis_dir = os.path.join(real_output_dir, "val_vis")
                        os.makedirs(vis_dir, exist_ok=True)
                        with torch.inference_mode(), temp_rng(args.val_seed or args.seed):
                            for i, batch in enumerate(val_dataloader):
                                if i >= args.val_vis_samples:
                                    break
                                out_file = os.path.join(vis_dir, f"step{global_step:08d}_sample{i:02d}.mp4")
                                visualize_val_batch(
                                    batch,
                                    out_file=out_file,
                                    infer_steps=args.val_vis_infer_steps,
                                    fps=args.val_vis_fps,
                                )
                                if args.wandb:
                                    wandb.log({f"val_vis/sample_{i}": wandb.Video(out_file, format="mp4")},step=global_step)
                gc.collect()
                torch.cuda.empty_cache()
                progress_bar.unpause()
                if use_ddp: dist.barrier()
            
            if global_step >= args.max_train_steps or global_step % args.checkpointing_steps == 0:
                if is_main:
                    save_file(get_peft_model_state_dict(M_ddp()),os.path.join(checkpoint_dir, f"wan-lora-{global_step:08}.safetensors"),
                    )
                    # patch embedding weights
                    pe = M_ddp().patch_embedding
                    torch.save({
                        "in_dim": M_ddp().in_dim,
                        "weight": pe.weight.detach().cpu(),
                        "bias":   (pe.bias.detach().cpu() if pe.bias is not None else None),
                    }, os.path.join(checkpoint_dir, f"patch_embedding-32-{global_step:08d}.pt"))
                    if args.residual_head and residual_head is not None:
                        torch.save(R_ddp().state_dict(), os.path.join(checkpoint_dir, f"residual_head-{global_step:08d}.pt"))
                    # === write 'last' bundle every checkpointing step ===
                    save_last_bundle(
                        checkpoint_dir=checkpoint_dir,
                        global_step=global_step,
                        diffusion_model=M_ddp(),
                        residual_head=R_ddp() if args.residual_head else None,
                        optimizer_dict=optimizer_dict,
                        lr_scheduler=None,
                        scaler=None
                    )
                gc.collect()
                torch.cuda.empty_cache()
                if use_ddp: dist.barrier()
            if global_step >= args.max_train_steps:
                break
    # === training finished: write 'last' bundle once ===
    if is_main:
        try:
            save_last_bundle(
                checkpoint_dir=checkpoint_dir,
                global_step=global_step,
                diffusion_model=M_ddp(),
                residual_head=R_ddp() if args.residual_head else None,
                optimizer_dict=optimizer_dict,
                lr_scheduler=None,
                scaler=None
            )
        except Exception as e:
            print(f"[last] save failed: {e}")

def parse_args():
    parser = argparse.ArgumentParser(
        description = "FADRA training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--download_model",
        action = "store_true",
        help = "auto download all necessary models to ./models if missing",
    )
    parser.add_argument(
        "--cache_embeddings",
        action = "store_true",
        help = "preprocess dataset to encode captions",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B",
        help="Path to pretrained model or model identifier from huggingface",
    )
    parser.add_argument(
        "--output_dir",
        type = str,
        default = "./outputs",
        help = "Output directory for training results",
    )
    parser.add_argument(
        "--dataset",
        type = str,
        default = None,
        help = "Path to dataset directory with train and val subdirectories",
    )
    parser.add_argument(
        "--latent_cache_dir",
        type = str,
        default = None,
        help = "Path to cached VAE latents (with manifest.json) for training; if set, skip VAE encoding for train",
    )
    parser.add_argument(
        "--latent_cache_val_dir",
        type = str,
        default = None,
        help = "Optional cached VAE latents dir for validation; if unset, validation encodes the raw videos of the val/test split",
    )
    parser.add_argument(
        "--train_batch_size",
        type = int,
        default = 32,
        help = "training batch size"
    )
    parser.add_argument(
        "--test_batch_size",
        type = int,
        default = 32,
        help = "test batch size"
    )


    parser.add_argument(
        "--val_samples",
        type = int,
        default = 4,
        help = "Maximum number of samples to use for validation loss",
    )
    parser.add_argument(
        "--seed",
        type = int,
        default = 42,
        help = "Seed for reproducible training",
    )
    parser.add_argument(
        "--val_seed",
        type = int,
        default = None,
        help = "Optional separate seed for validation, to have consistent validation rng when changing training seed",
    )
    parser.add_argument(
        "--lora_targets",
        type = str,
        default = "self_attn, cross_attn",
        help = "layers to target with LoRA",
    )
    parser.add_argument(
        "--lora_blocks",
        type = str,
        default = None,
        help = "blocks to target with LoRA, default is all blocks, otherwise comma separated string of block numbers",
    )
    parser.add_argument(
        "--control_lora",
        action = "store_true",
        help = "Train lora as control lora (extra input channels)",
    )
    parser.add_argument(
        "--residual_head",
        action = "store_true",
        help = "Enable a small residual head to correct predicted v",
    )
    parser.add_argument(
        "--residual_input",
        type = str,
        default = "pred+control",
        choices = ["pred", "control", "pred+control"],
        help = "Input for residual head: predicted v, control latents, or concatenation",
    )
    parser.add_argument(
        "--residual_layers",
        type = int,
        default = 1,
        help = "Number of DiT blocks in residual head (deprecated; use --residual_dit_blocks)",
    )
    parser.add_argument(
        "--residual_dit_blocks",
        type = int,
        default = None,
        help = "Alias for --residual_layers (DiT blocks in residual head)",
    )
    parser.add_argument(
        "--residual_lr",
        type = float,
        default = None,
        help = "LR for residual head (defaults to --learning_rate if unset)",
    )
    parser.add_argument(
        "--lora_rank",
        type = int,
        default = 16,
        help = "The dimension of the LoRA update matrices",
    )
    parser.add_argument(
        "--lora_alpha",
        type = int,
        default = None,
        help = "The alpha value for LoRA, defaults to alpha=rank. Note: changing alpha will affect the learning rate, and if alpha=rank then changing rank will also affect learning rate",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action = "store_true",
        help = "use gradient checkpointing to reduce memory usage at the cost of speed",
    )
    parser.add_argument(
        "--learning_rate",
        type = float,
        default = 1e-5,
        help = "Base learning rate",
    )
    parser.add_argument(
        "--input_lr_scale",
        type = float,
        default = 1,
        help = "Multiplier to learning rate for the input patch_embedding layer if training control lora",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default = 3.0,
        help = "Noise schedule shift for training (shift > 1 will spend more effort on early timesteps/high noise",
    )
    parser.add_argument(
        "--distill_negative",
        type = str,
        default = "./embeddings/default_video_negative_wan.safetensors",
        help = "Precalculated negative prompt embedding (used as the unconditional context in validation visualization)",
    )
    parser.add_argument(
        "--base_res",
        type = int,
        nargs = '+',
        default = [624],
        choices=[624, 960, 1024],
        help = "Base resolution buckets, resized to equal area based on aspect ratio. Supports multiple buckets.",
    )
    parser.add_argument(
        "--token_limit",
        type = int,
        default = 10_000,
        help = "Combined resolution/frame limit based on transformer patch sequence length: (width // 16) * (height // 16) * ((frames - 1) // 4 + 1)",
    )
    parser.add_argument(
        "--max_frame_stride",
        type = int,
        default = 2,
        help = "1: use native framerate only. Higher values allow randomly choosing lower framerates (skipping frames to speed up the video)",
    )
    parser.add_argument(
        "--random_frame_length",
        action = "store_true",
        help = "Randomize frame length instead of always trying to fill the max length",
    )
    parser.add_argument(
        "--val_steps",
        type = int,
        default = 50,
        help = "Validate after every n steps",
    )
    parser.add_argument(
        "--val_timestep",
        type = int,
        default = None,
        help = "Override timestep for validation samples",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type = int,
        default = 100,
        help = "Save a checkpoint of the training state every X steps",
    )
    parser.add_argument(
        "--max_train_steps",
        type = int,
        default = 5,
        help = "Total number of training steps",
    )
    
    # --- wandb options ---
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="FADRA",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (team/user). None uses your default.")
    parser.add_argument("--wandb_run_name", type=str, default="FADRA",
                        help="Optional custom run name")
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="Optional group name (e.g., dataset/exp tag)")
    parser.add_argument("--wandb_tags", type=str, default="fadra",
                        help="Comma-separated tags, e.g. 'wan,lora,control'")
    # ---- validation visualization options ----
    parser.add_argument(
        "--val_vis",
        action="store_true",
        help="Enable visualizing validation samples (save side-by-side videos).",
    )
    parser.add_argument(
        "--val_vis_samples",
        type=int,
        default=1,
        help="How many samples to visualize each validation.",
    )
    parser.add_argument(
        "--val_vis_infer_steps",
        type=int,
       default=30,
        help="Inference steps for visualization sampler.",
    )
    parser.add_argument(
        "--val_vis_fps",
        type=int,
        default=16,
        help="FPS for saved validation videos.",
    )
    # ----- resume options -----
    parser.add_argument("--resume_dir", type=str, default=None,
                        help="Resume into an existing output run directory (./outputs/xxx).")
    parser.add_argument("--resume_wandb_id", type=str, default=None,
                        help="If using W&B, resume the same run id.")
    parser.add_argument("--pe_ckpt_path", type=str, default=None,
                        help="Path to patch embedding checkpoint to resume.")
    parser.add_argument("--use_freq_loss",
        action = "store_true",
        help = "Use frequency domain loss in addition to spatial MSE loss",
    )
    parser.add_argument("--freq_loss_weight",
        type = float,
        default = 1.0,
        help = "Weight for frequency domain loss",
    )
                        

    args = parser.parse_args()
    if args.residual_dit_blocks is not None:
        args.residual_layers = args.residual_dit_blocks
    return args


if __name__ == "__main__":
    args = parse_args()
    
    if args.download_model:
        download_model(args)
        exit()
    
    if args.dataset is not None:
        if args.cache_embeddings:
            cache_embeddings(args)
            exit()
        
        main(args)
    else:
        raise Exception("--dataset is required but not provided")
