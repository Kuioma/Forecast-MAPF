
import multiprocessing
import sys
import os
import time
import math
import dataclasses
from dataclasses import dataclass, field, asdict
from contextlib import nullcontext
from ast import literal_eval
from typing import Optional, Tuple, Any, Dict

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
wandb.init(mode="offline")
from loguru import logger

# Add project root to path
sys.path.append("/home/mapf-gpt")

# Internal imports
from gpt.fast_data_loader import MapfArrowDatasetMultiAction
from gpt.model_multi_action import GPT, GPTConfig
from tokenizer.parameters import InputParameters
from tokenizer.tokenizer import Tokenizer

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    # I/O
    out_dir: str = "four_action/21/6m"
    eval_interval: int = 500
    log_interval: int = 1
    eval_iters: int = 5
    always_save_checkpoint: bool = True
    init_from: str = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
    
    # wandb logging
    wandb_log: bool = True
    wandb_project: str = "mapf-gpt-multi-action"
    
    # Training Loop
    gradient_accumulation_steps: int = 16
    batch_size: int = 32
    block_size: int = 256
    
    # Model
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False
    morden_style: bool = True
    action_mask: bool = True
    
    # Optimizer
    learning_rate: float = 6e-4
    max_iters: int = 30000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # LR Decay
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 30000
    min_lr: float = 6e-5
    
    # DDP
    backend: str = "nccl"
    device: str = "cuda"  # 'cpu', 'cuda', 'cuda:0', 'mps'
    
    # Type validity
    dtype: str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    compile: bool = True
    
    # Multi-action settings
    num_actions: int = 4
    train_data_file: str = "dataset/1"
    valid_data_file: str = "dataset/1"
    
    def update_from_args(self, args: list[str]):
        """
        Parses command line arguments to update config values.
        Supports:
          script.py config_file.py
          script.py --key=value
        """
        initial_config_keys = {k for k in asdict(self).keys()}
        
        for arg in args:
            if '=' not in arg:
                # Assume it's a config file
                config_file = arg
                logger.info(f"Overriding config with {config_file}")
                # We execute the file and capture variables
                file_globals = {}
                try:
                    with open(config_file) as f:
                        exec(f.read(), {}, file_globals)
                    
                    for k, v in file_globals.items():
                        if k in initial_config_keys:
                            if isinstance(v, type(getattr(self, k))):
                                setattr(self, k, v)
                                logger.info(f"Overriding from file: {k} = {v}")
                            else:
                                logger.warning(f"Type mismatch for {k}: expected {type(getattr(self, k))}, got {type(v)}")
                except Exception as e:
                    logger.error(f"Failed to load config file {config_file}: {e}")
                    raise
            else:
                # Assume --key=value
                if not arg.startswith('--'):
                    logger.warning(f"Argument {arg} ignored (expected --key=value)")
                    continue
                
                key, val = arg.split('=', 1)
                key = key[2:]
                
                if hasattr(self, key):
                    current_val = getattr(self, key)
                    try:
                        attempt = literal_eval(val)
                    except (SyntaxError, ValueError):
                        attempt = val
                    
                    # Ensure minimal type sanity (rough check)
                    # Note: Optional fields might need care, but here most are strict types
                    if type(attempt) == type(current_val):
                         setattr(self, key, attempt)
                         logger.info(f"Overriding CLI: {key} = {attempt}")
                    else:
                        # Allow int -> float conversions
                        if isinstance(current_val, float) and isinstance(attempt, int):
                            setattr(self, key, float(attempt))
                            logger.info(f"Overriding CLI: {key} = {float(attempt)}")
                        else:
                             # For strings that literal_eval didn't parse as something else, we might be fine
                             if isinstance(current_val, str):
                                 setattr(self, key, str(attempt))
                                 logger.info(f"Overriding CLI: {key} = {attempt}")
                             else:
                                 logger.warning(f"Type mismatch for {key}: expected {type(current_val)}, got {type(attempt)}")
                else:
                    logger.warning(f"Unknown config key: {key}")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def calculate_epochs(max_iters, dataset_size, batch_size, gradient_accumulation_steps=1):
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = dataset_size // effective_batch_size
    # Avoid div by zero if dataset is small
    if steps_per_epoch == 0:
        return 0
    num_epochs = max_iters / steps_per_epoch
    return num_epochs

def human_readable_size(size):
    for unit in ["pairs", "K pairs", "M pairs", "B pairs"]:
        if size < 1000:
            return f"{size:.2f} {unit}"
        size /= 1000
    return f"{size:.2f} B pairs"

def get_batch(data_iter, device):
    x, y = next(data_iter)
    # Move to device here to avoid doing it inside the loop manually every time if possible,
    # though original code did it inside config logic.
    # The original code did: x.to(torch.int), y.to(torch.long) then model.to(device).
    # Data loader usually returns cpu tensors.
    if device is not None:
        if isinstance(device, str):
             x = x.to(device)
             y = y.to(device)
        else:
             # If device is something else
             pass
    return x.to(torch.int), y.to(torch.long)

@torch.no_grad()
def estimate_loss(model, ctx, train_iter, val_iter, eval_iters):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            iterator = train_iter if split == "train" else val_iter
            # Note: get_batch in original code relies on a global `get_batch` which relies on `next(data)`.
            # We need to make sure we don't exhaust the iterator destructively if that matters.
            # But typically for training we just pull next. For eval, strict correctness might imply separate iterator.
            # Original code reused `train_data_iter` and `val_data_iter`.
            # Wait, `get_batch` calls `next(data)`.
            
            # Use the global `device`? We need to pass device to get_batch.
            # We will infer device from model usage or pass it.
            # Let's assume model is on correct device, so inputs need to be on that device.
            device = next(model.parameters()).device
            X, Y = get_batch(iterator, device)
            
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it, config: TrainingConfig):
    # 1) linear warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # 2) > lr_decay_iters -> min_lr
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

# -----------------------------------------------------------------------------
# Main Training Logic
# -----------------------------------------------------------------------------

def setup_ddp(config: TrainingConfig):
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend=config.backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert config.gradient_accumulation_steps % ddp_world_size == 0
        config.gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = config.device
        
        # Check cuda availability
        if 'cuda' in device and not torch.cuda.is_available():
            logger.warning('Cuda is not available, switching to cpu')
            device = 'cpu'
            config.device = 'cpu'
            
    return ddp, device, master_process, seed_offset, ddp_world_size

def train(config: TrainingConfig):
    # DDP Setup
    ddp, device, master_process, seed_offset, ddp_world_size = setup_ddp(config)
    
    # Logging
    if master_process:
        os.makedirs(config.out_dir, exist_ok=True)
        if config.wandb_log:
            wandb.init(project=config.wandb_project, config=asdict(config), mode="offline" if not config.wandb_log else "online")
            # Note: The original code forced "offline" separately on line 13. 
            # I'll respect the config.wandb_log but also check if user wanted offline generally?
            # Original code had `wandb.init(mode="offline")` at import time!
            # We should probably respect that if it was intentional.
            
    # Seeds
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Context
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Data
    if master_process:
        logger.info("Initializing Datasets...")
    train_data = MapfArrowDatasetMultiAction(config.train_data_file, device=device, batch_size=config.batch_size, action_n=config.num_actions)
    val_data = MapfArrowDatasetMultiAction(config.valid_data_file, device=device, batch_size=config.batch_size, action_n=config.num_actions)
    
    train_data_iter = iter(train_data)
    val_data_iter = iter(val_data)
    
    if master_process:
        logger.info(f"Train set size: {human_readable_size(train_data.get_full_dataset_size())}")
        logger.info(f"Validation set size: {human_readable_size(val_data.get_full_dataset_size())}")
        num_epochs = calculate_epochs(config.max_iters, train_data.get_full_dataset_size(), config.batch_size, config.gradient_accumulation_steps)
        logger.info(f"Number of training epochs: {num_epochs:.2f}")
    
    # Tokenizer
    tokenizer_cfg = InputParameters()
    tokenizer = Tokenizer(tokenizer_cfg)
    meta_vocab_size = len(tokenizer.encoder.vocab)
    
    # Model Init
    iter_num = 0
    best_val_loss = 1e9
    
    model_args = dict(
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        bias=config.bias,
        vocab_size=None,
        dropout=config.dropout,
        n_actions=config.num_actions,
        morden_style=config.morden_style,
        action_mask=config.action_mask
    )
    
    if config.init_from == "scratch":
        if master_process:
            logger.info("Initializing a new model from scratch")
        model_args["vocab_size"] = meta_vocab_size
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif config.init_from == "resume":
        if master_process:
            logger.info(f"Resuming training from {config.out_dir}")
        ckpt_path = os.path.join(config.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        # Force config attributes equality
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size", "num_actions"]:
            model_args[k] = checkpoint_model_args[k]
        
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    
    model.to(device)
    
    if master_process:
        logger.info("number of parameters: %.2fM" % (model.get_num_params() / 1e6,))
        
    scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == "float16"))
    optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type)
    
    if config.init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    # Compilation
    if config.compile and 'cuda' in device:
        if master_process:
            logger.info("compiling the model... (takes a ~minute)")
        try:
            model = torch.compile(model)
        except AttributeError:
            logger.warning('torch compile(model) requires PyTorch >= 2.0')
            
    # DDP Wrap
    if ddp:
        model = DDP(model, device_ids=[int(device.split(':')[1])])
        
    raw_model = model.module if ddp else model
    
    # Training Loop
    X, Y = get_batch(train_data_iter, device)
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0
    
    while True:
        # LR Schedule
        lr = get_lr(iter_num, config) if config.decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
        # Evaluation and Checkpointing
        if iter_num % config.eval_interval == 0 and master_process:
            losses = estimate_loss(model, ctx, train_data_iter, val_data_iter, config.eval_iters)
            logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            if config.wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,
                })
                
            if losses["val"] < best_val_loss or config.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": asdict(config),
                    }
                    logger.info(f"saving checkpoint to {config.out_dir}")
                    torch.save(checkpoint, os.path.join(config.out_dir, "ckpt.pt"))

        # Training Step
        for micro_step in range(config.gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == config.gradient_accumulation_steps - 1)
                
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / config.gradient_accumulation_steps
                
            X, Y = get_batch(train_data_iter, device)
            scaler.scale(loss).backward()
            
        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Timing
        t1 = time.monotonic()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % config.log_interval == 0 and master_process:
            lossf = loss.item() * config.gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(config.batch_size * config.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                
            logger.info(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, "
                        f"mfu {running_mfu * 100:.2f}%, {iter_num} / {config.max_iters}")
            
        iter_num += 1
        local_iter_num += 1
        
        if iter_num > config.max_iters:
            break
            
    if ddp:
        destroy_process_group()

def main():
    # Ensure spawn for multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    
    # Wandb Offline Default Init (as per original script being conservative)
    # However, init happens later in train() properly based on config.
    # The original script had it early.
    # wandb.init(mode="offline") 
    
    config = TrainingConfig()
    config.update_from_args(sys.argv[1:])
    
    train(config)

if __name__ == "__main__":
    main()