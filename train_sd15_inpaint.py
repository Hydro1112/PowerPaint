#!/usr/bin/env python
# Finetune Stable Diffusion 1.5 Inpainting on COCO2017 with on-the-fly masks.
# Adapted from train_ppt2_bn.py (PowerPaint/BrushNet training loop) to keep the
# same project structure: COCODataset mask generation, validation loss on a
# held-out split, checkpoint resume, and best-model export.

import argparse
import gc
import json
import logging
import math
import os
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from powerpaint.datasets import COCODataset


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

logger = get_logger(__name__)


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]

            validation_image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# SD1.5 Inpainting - {repo_id}

These are SD 1.5 Inpainting weights trained on COCO2017 with on-the-fly masks.
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )
    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "inpainting",
        "diffusers",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(unet, args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    # Reload the full pipeline with the trained UNet. The UNet needs to be
    # wrapped because of the distributed setup.
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipe = pipe.to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()

    # load validation images
    image_logs = []
    for case in args.validation_data.cases:
        validation_prompts = case.prompt
        validation_image = Image.open(os.path.join(args.validation_data.data_root, case.image)).convert("RGB")
        # COCODataset convention: mask 1 == masked (hole). The inpaint pipeline
        # uses the opposite (white == repainted), so invert here.
        validation_mask = Image.open(os.path.join(args.validation_data.data_root, case.mask))
        validation_mask = validation_mask.resize(
            (validation_image.size[0], validation_image.size[1]), Image.NEAREST
        ).convert("L")
        validation_mask = validation_mask.point(lambda p: 255 - p)

        image_grid = Image.new(
            "RGB",
            (validation_image.size[0] * (1 + len(validation_prompts)), validation_image.size[1]),
            (255, 255, 255),
        )
        image_grid.paste(validation_image, (0, 0))
        for i, p in enumerate(validation_prompts):
            with torch.autocast(accelerator.device.type):
                image = pipe(
                    prompt=p.prompt,
                    negative_prompt=p.negative_prompt,
                    image=validation_image,
                    mask_image=validation_mask,
                    num_inference_steps=20,
                ).images[0]
            image_logs.append(image)
            image_grid.paste(image, (validation_image.size[0] * (i + 1), 0))
        image_grid.save(os.path.join(args.output_dir, f"{str(step).zfill(3)}_{os.path.basename(case.image)}"))
    gc.collect()
    torch.cuda.empty_cache()

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in image_logs])
            tracker.writer.add_images("validation", np_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{p.task}")
                        for image, p in zip(image_logs, args.validation_data.cases[0].prompt)
                    ]
                }
            )
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return image_logs


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Simple example of a Stable Diffusion 1.5 Inpainting training script on COCO."
    )
    parser.add_argument("--config", type=str, default=None, help="yaml for configuration")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/sd15_inpaint",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--revision", type=str, default=None, required=False)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=10000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.',
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Initial learning rate (after the potential warm up period) to use.")
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between fp16 and bf16. Bf16 requires PyTorch >= 1.10 and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--proportion_empty_prompts", type=float, default=0)
    parser.add_argument("--snr_gamma", type=float, default=None, help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0.")
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help="Run validation every X steps. Validation consists of computing the held-out loss and generating images.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_sd15_inpaint",
        help="The `project_name` argument passed to Accelerator.init_trackers.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # use omegaconf to manage configurations
    if args.config is not None:
        config = OmegaConf.load(args.config)
        # CLI always wins over config: only apply config keys whose value is still the parser default.
        try:
            default_args = parser.parse_args([])
        except SystemExit:
            default_args = None
        if default_args is not None:
            for k, v in config.items():
                if getattr(args, k, None) == getattr(default_args, k, None):
                    args.__dict__[k] = v

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the UNet encoder."
        )

    return args


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        torch.manual_seed(args.seed)
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # saving training configuration to output_dir
        to_save_config = OmegaConf.create(vars(args))
        OmegaConf.save(config=to_save_config, f=os.path.join(args.output_dir, "training_config.yaml"))

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load the tokenizer, text encoder, VAE, and UNet from the SD1.5 inpainting
    # pipeline. The UNet is loaded separately in full float32 precision (even
    # under mixed precision) so it can be trained; only vae/text_encoder are
    # cast to weight_dtype for inference.
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        ),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    vae, tokenizer, unet, noise_scheduler = pipe.vae, pipe.tokenizer, pipe.unet, pipe.scheduler
    text_encoder = pipe.text_encoder.to(torch.float32)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for model in models:
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )
    if unwrap_model(unet).dtype != torch.float32:
        raise ValueError(f"UNet loaded as datatype {unwrap_model(unet).dtype}. {low_precision_error_string}")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # 1. trainable UNet; VAE and text encoder are frozen.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(True)

    optimizer = optimizer_class(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # transforms used for preprocessing dataset
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.resolution, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # The COCODataset returns samples with `mask` (1 == masked) and `input_ids`.
    # COCODataset samples a task key internally and reads
    # `task_prompt[task_key].placeholder_tokens`; use empty placeholder tokens so
    # the caption is passed through unchanged (no learnable tokens for SD1.5 inpaint).
    task_prompt = OmegaConf.create(
        {
            "text_guided_object_synthesis": {"placeholder_tokens": ""},
            "object_removal": {"placeholder_tokens": ""},
        }
    )

    logger.info("Loading COCO train dataset (this loads instances annotation, may take a minute)...")
    train_dataset = COCODataset(
        train_transforms,
        pipe,
        task_prompt,
        **args.train_data.datasets[0],
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # `validation_dataset` is a held-out dataset used to compute diffusion
    # validation loss. It is intentionally separate from `validation_data`,
    # which is only used below for qualitative image generation.
    validation_dataloader = None
    if hasattr(args, "validation_dataset"):
        logger.info("Loading COCO validation dataset...")
        validation_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        validation_dataset = COCODataset(
            validation_transforms,
            pipe,
            task_prompt,
            is_validation=True,
            **args.validation_dataset,
        )
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=getattr(args, "validation_batch_size", args.train_batch_size),
            shuffle=False,
            num_workers=args.dataloader_num_workers,
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    unet.train()
    # Prepare everything with our `accelerator`.
    if validation_dataloader is not None:
        unet, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, validation_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        pop_list = []
        for k, v in tracker_config.items():
            if not isinstance(v, (int, float, str, bool, torch.Tensor)):
                pop_list.append(k)
                logger.info(f"Removed {k} (type:{type(v)}) from tracker_config")
        for k in pop_list:
            tracker_config.pop(k)

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info(f"***** Running training for {args.tracker_project_name} *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {int(args.max_train_steps)}")
    if validation_dataloader is not None:
        logger.info(f"  Validation examples = {len(validation_dataset)}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path), map_location="cpu")
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # Track the best validation loss and export the corresponding weights.
    best_validation_loss = float("inf")
    best_ckpt_step = None
    best_ckpt_name = None
    best_ckpt_dir = os.path.join(args.output_dir, "best_model")

    # Restore best-checkpoint state from a previous run so that resuming keeps
    # the historical best instead of resetting it.
    if accelerator.is_main_process and args.resume_from_checkpoint:
        validation_log_path = os.path.join(args.output_dir, "validation_log.csv")
        if os.path.exists(validation_log_path):
            with open(validation_log_path, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("step")]
            best_line = min(lines, key=lambda ln: float(ln.split(",")[1]), default=None) if lines else None
            if best_line is not None:
                best_step, best_loss = best_line.split(",")[0], float(best_line.split(",")[1])
                best_validation_loss = best_loss
                best_ckpt_step = int(best_step)
                best_ckpt_name = f"checkpoint-{best_step}"
                logger.info(
                    f"Resumed best-checkpoint state: best_validation_loss={best_validation_loss:.6f} "
                    f"at step {best_ckpt_step}"
                )

    progress_bar = tqdm(
        range(0, int(args.max_train_steps)),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    log_interval = 10

    if accelerator.is_main_process:
        loss_log_path = os.path.join(args.output_dir, "training_log.csv")
        loss_file = open(loss_log_path, "w")
        loss_file.write("step,epoch,batch_accum_loss,train_loss,lr,grad_norm\n")

        validation_log_path = os.path.join(args.output_dir, "validation_log.csv")
        validation_file = open(validation_log_path, "w")
        validation_file.write("step,validation_loss\n")

    def _export_model(export_dir, source_unet):
        """Export the full runnable artifact: a standalone SD1.5 inpaint pipeline
        with the trained UNet swapped in (diffusers-style folder layout)."""
        os.makedirs(export_dir, exist_ok=True)

        # Save the trained UNet in diffusers format (in_channels=9 -> inpainting).
        source_unet.save_pretrained(os.path.join(export_dir, "unet"))

        # Copy the remaining components from the base pipeline (tokenizer,
        # text_encoder, vae, scheduler) so the folder is a full pipeline.
        for sub in ["tokenizer", "text_encoder", "vae", "scheduler"]:
            src = os.path.join(args.pretrained_model_name_or_path, sub)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(export_dir, sub), dirs_exist_ok=True)

        # Manifest: base model + best checkpoint needed for inference.
        manifest = {
            "base_model_name_or_path": args.pretrained_model_name_or_path,
            "best_checkpoint": best_ckpt_name,
            "best_validation_loss": best_validation_loss,
            "best_step": best_ckpt_step,
        }
        with open(os.path.join(export_dir, "inference_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    @torch.no_grad()
    def run_validation_loss():
        """Evaluate the held-out set with the same diffusion objective as training.

        Noise and timesteps are drawn from a generator seeded by global_step so
        the reported loss is reproducible between runs (dataset masks are already
        seeded by sample idx inside COCODataset for validation).
        """
        val_generator = torch.Generator(device=accelerator.device).manual_seed(global_step)
        unet.eval()
        total_loss = torch.zeros((), device=accelerator.device)
        total_batches = torch.zeros((), device=accelerator.device)
        logger.info("Running validation loss over %d samples...", len(validation_dataset))

        for batch in validation_dataloader:
            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
            latents = latents * vae.config.scaling_factor

            # mask: 1 for masked regions, 0 for known regions (COCODataset convention).
            mask = torch.nn.functional.interpolate(batch["mask"], size=(64, 64))
            masked_image = batch["pixel_values"] * (batch["mask"] < 0.5)
            masked_image = masked_image - batch["mask"]
            masked_latents = vae.encode(masked_image.to(dtype=weight_dtype)).latent_dist.sample()
            masked_latents = (masked_latents * vae.config.scaling_factor).to(weight_dtype)

            noise = torch.randn(latents.shape, generator=val_generator)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device,
                generator=val_generator,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)
            encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

            model_pred = unet(
                latent_model_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states.detach().to(weight_dtype),
                return_dict=False,
            )[0]
            target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, timesteps)
            if noise_scheduler.config.prediction_type not in {"epsilon", "v_prediction"}:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            if args.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                snr = compute_snr(noise_scheduler, timesteps)
                weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                weights = weights / (snr if noise_scheduler.config.prediction_type == "epsilon" else snr + 1)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = (loss.mean(dim=list(range(1, len(loss.shape)))) * weights).mean()
            total_loss += loss.detach()
            total_batches += 1

        totals = accelerator.gather(torch.stack([total_loss, total_batches])).reshape(-1, 2).sum(dim=0)
        validation_loss = (totals[0] / totals[1]).item()
        unet.train()
        return validation_loss

    logger.info("Training started. Progress lines are printed every %d steps.", log_interval)
    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info("===== Starting epoch %d/%d =====", epoch + 1, args.num_train_epochs)
        train_loss = 0.0
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor

                # mask: 1 for masked regions, 0 for known regions.
                mask = torch.nn.functional.interpolate(batch["mask"], size=(64, 64))
                masked_image = batch["pixel_values"] * (batch["mask"] < 0.5)
                # convert the hole value from 0 to -1 due to [-1, 1] range
                masked_image = masked_image - batch["mask"]
                masked_latents = vae.encode(masked_image.to(dtype=weight_dtype)).latent_dist.sample()
                masked_latents = (masked_latents * vae.config.scaling_factor).to(weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # concatenate the noised latents with the mask and the masked latents
                latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                # Predict the noise residual
                model_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states.detach().to(weight_dtype),
                    return_dict=False,
                )[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm).item()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)

                # Print a clear progress line every `log_interval` steps so the
                # notebook shows live training progress instead of silence.
                if accelerator.is_main_process and global_step % log_interval == 0:
                    cur_lr = lr_scheduler.get_last_lr()[0]
                    eta_steps = int(args.max_train_steps) - global_step
                    logger.info(
                        f"[step {global_step}/{int(args.max_train_steps)}] "
                        f"epoch {epoch} | train_loss {train_loss:.6f} | "
                        f"step_loss {loss.detach().item():.6f} | lr {cur_lr:.2e} | "
                        f"grad_norm {grad_norm:.4f} | eta_steps {eta_steps}"
                    )
                    loss_file.write(f"{global_step},{epoch},{loss.detach().item()},{train_loss},{cur_lr},{grad_norm}\n")
                    loss_file.flush()

                train_loss = 0.0

                if validation_dataloader is not None and global_step % args.validation_steps == 0:
                    validation_loss = run_validation_loss()
                    accelerator.log({"validation_loss": validation_loss}, step=global_step)
                    if accelerator.is_main_process:
                        logger.info(f"Validation loss at step {global_step}: {validation_loss:.6f}")
                        validation_file.write(f"{global_step},{validation_loss}\n")
                        validation_file.flush()

                        # Save the best checkpoint based on held-out loss.
                        if validation_loss < best_validation_loss:
                            best_validation_loss = validation_loss
                            best_ckpt_step = global_step
                            best_ckpt_name = f"checkpoint-{global_step}"
                            logger.info(
                                f"New best validation loss {validation_loss:.6f} at step "
                                f"{global_step}; exporting best_model/"
                            )
                            if os.path.isdir(best_ckpt_dir):
                                shutil.rmtree(best_ckpt_dir)
                            _export_model(best_ckpt_dir, accelerator.unwrap_model(unet))

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if hasattr(args, "validation_data") and global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Evaluate once more after the final optimizer step.  Every process must
    # participate because run_validation_loss gathers values across processes.
    if validation_dataloader is not None:
        validation_loss = run_validation_loss()
        accelerator.log({"validation_loss": validation_loss}, step=global_step)
        if accelerator.is_main_process:
            logger.info(f"Final validation loss: {validation_loss:.6f}")
            validation_file.write(f"{global_step},{validation_loss}\n")
            validation_file.flush()

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_ckpt_step = global_step
                best_ckpt_name = f"checkpoint-{global_step}"
                logger.info(
                    f"New best validation loss {validation_loss:.6f} at step {global_step}; "
                    f"exporting best_model/"
                )
                if os.path.isdir(best_ckpt_dir):
                    shutil.rmtree(best_ckpt_dir)
                _export_model(best_ckpt_dir, accelerator.unwrap_model(unet))

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if validation_dataloader is not None:
            validation_file.close()
        loss_file.close()

        unet = unwrap_model(unet)
        # Full artifact for the final step.
        _export_model(args.output_dir, unet)

        # Report the best checkpoint for inference.
        if best_ckpt_step is not None:
            logger.info(
                f"Best validation loss {best_validation_loss:.6f} at step {best_ckpt_step} "
                f"-> {best_ckpt_dir} (use this for inference)"
            )
        else:
            logger.info("No validation ran; output_models/ is the final-step artifact")

        # Run a final round of validation.
        image_logs = None
        if hasattr(args, "validation_data"):
            image_logs = log_validation(
                unet,
                args,
                accelerator,
                weight_dtype,
                global_step,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
