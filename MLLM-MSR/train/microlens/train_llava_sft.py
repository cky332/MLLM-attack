import os
from datetime import timedelta
from lightning.pytorch.strategies import DeepSpeedStrategy
import time

os.environ["NCCL_DEBUG"] = "INFO"  # Enable NCCL debug output
os.environ["NCCL_TIMEOUT"] = "1800000"  # 30 minutes in ms (default is 600s=10min)

MAX_LENGTH = 2048
EPOCH = 4
LORA_R = 16
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
REPO_ID = "yeyuyang95/llava-v1.6-mistral-7b-hf-lora"
WANDB_PROJECT = "LLaVaNeXT"
WANDB_NAME = "llava-v1.6-mistral-7b-hf-lora"
SAVE_DIR = f"/home/chenkuiyun/MLLM/output/llava-v1.6-mistral-7b-hf-lora-recurrent-e{EPOCH}-r{LORA_R}"
LOG_DIR = "/home/chenkuiyun/MLLM/output/logs"

from transformers import AutoProcessor
from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from typing import Any, Dict
import random
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from load_llava_dataset import LlavaDataset, LlavaDataset2
from PIL import ImageOps
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from torch.utils.data import DataLoader
import re
import os
from nltk import edit_distance
import numpy as np
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import logging

from huggingface_hub import HfApi

logging.getLogger("transformers").setLevel(logging.ERROR)

# ============================================================
# Debug Logger Setup — each rank writes to its own log file
# ============================================================
os.makedirs(LOG_DIR, exist_ok=True)

def setup_debug_logger():
    """Create a per-rank file logger. Call after distributed init."""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    logger = logging.getLogger("debug_train")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(
            os.path.join(LOG_DIR, f"training_debug_rank{rank}.log"),
            mode="w",
        )
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            f"[Rank {rank}] %(asctime)s.%(msecs)03d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def get_gpu_mem_mb():
    """Return (allocated_MB, reserved_MB, free_MB) for current device."""
    if not torch.cuda.is_available():
        return 0, 0, 0
    dev = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(dev) / 1024**2
    reserved = torch.cuda.memory_reserved(dev) / 1024**2
    total = torch.cuda.get_device_properties(dev).total_memory / 1024**2
    free = total - reserved
    return alloc, reserved, free


processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

USE_LORA = True
USE_QLORA = False

## Load model

# Three options for training, from the lowest precision training to the highest precision training:
# - QLora
# - Standard Lora
# - Full fine-tuning
if USE_QLORA or USE_LORA:
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )

        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID,
            cache_dir=os.path.expanduser('~/.cache/huggingface/hub'),
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
        )
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID,
            cache_dir=os.path.expanduser('~/.cache/huggingface/hub'),
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
        )
else:
    # for full fine-tuning, we can speed up the model using Flash Attention
    # only available on certain devices, see https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir='/data1/share/.HF_cache/',
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
    )


#Apply PEFT
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

print(find_all_linear_names(model))


lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model = get_peft_model(model, lora_config)

#Create PyTorch dataset
train_dataset = LlavaDataset2("MicroLens-50k-training-recurrent",  split="train", sort_json_key=False)
val_dataset = LlavaDataset2("MicroLens-50k-training-recurrent", split="validation", sort_json_key=False)

def resize_image(image_list):
    max_width = max(img.width for img in image_list)
    max_height = max(img.height for img in image_list)

    padded_images = []
    for img in image_list:
        if img.width == max_width and img.height == max_height:
            padded_images.append(img)
            continue
        else:
            delta_width = max_width - img.width
            delta_height = max_height - img.height

            padding = (
                delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))

            new_img = ImageOps.expand(img, border=padding, fill='black')
            padded_images.append(new_img)

    return padded_images

def train_collate_fn(examples):
    dlog = logging.getLogger("debug_train")
    t0 = time.time()
    images = []
    texts = []
    for example in examples:
        image, prompt_text, ground_truth = example
        images.append(image)
        prompt = f"[INST] <image>\n{prompt_text} [\INST] {ground_truth}"
        texts.append(prompt)
    images = resize_image(images)

    batch = processor(text=texts, images=images, padding=True, truncation=False, max_length=MAX_LENGTH, return_tensors="pt")

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    image_sizes = batch["image_sizes"]
    labels = batch["labels"]

    elapsed = time.time() - t0
    if elapsed > 5.0:  # Only log slow collates (> 5s)
        dlog.warning(f"SLOW collate_fn took {elapsed:.2f}s | seq_len={input_ids.shape[-1]} | img_sizes={image_sizes.tolist()}")

    return input_ids, attention_mask, pixel_values, image_sizes, labels

def eval_collate_fn(examples):
    images = []
    texts = []
    answers = []
    for example in examples:
        image, prompt_text, ground_truth = example
        images.append(image)
        prompt = f"[INST] <image>\n{prompt_text} [\INST]"
        texts.append(prompt)
        answers.append(ground_truth)
    images = resize_image(images)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    image_sizes = batch["image_sizes"]

    return input_ids, attention_mask, pixel_values, image_sizes, answers

#Define PyTorch LightningModule
class LlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.batch_size = config.get("batch_size")
        self._step_t0 = None
        self._dlog = None

    def _get_dlog(self):
        if self._dlog is None:
            self._dlog = setup_debug_logger()
        return self._dlog

    def on_train_start(self):
        dlog = self._get_dlog()
        dlog.info(f"=== TRAINING START === device={self.device} global_rank={self.global_rank}")
        dlog.info(f"Train dataset size: {len(train_dataset)}")
        alloc, reserved, free = get_gpu_mem_mb()
        dlog.info(f"GPU mem at train start: alloc={alloc:.0f}MB reserved={reserved:.0f}MB free={free:.0f}MB")

    def training_step(self, batch, batch_idx):
        dlog = self._get_dlog()
        step_start = time.time()

        input_ids, attention_mask, pixel_values, image_sizes, labels = batch
        seq_len = input_ids.shape[-1]

        self.model.train()
        outputs = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_sizes=image_sizes,
                            labels=labels
                          )
        loss = outputs.loss
        batch_size = input_ids.size(0)

        self.log("train_loss", loss, batch_size=batch_size)

        step_elapsed = time.time() - step_start
        alloc, reserved, free = get_gpu_mem_mb()

        # Log every 500 steps + last 50 steps of estimated epoch
        if batch_idx % 500 == 0 or batch_idx >= 9800:
            dlog.info(
                f"step={batch_idx} | loss={loss.item():.4f} | seq_len={seq_len} "
                f"| step_time={step_elapsed:.2f}s | gpu_alloc={alloc:.0f}MB "
                f"| gpu_free={free:.0f}MB | img_sizes={image_sizes.tolist()}"
            )

        # Alert on slow steps
        if step_elapsed > 30.0:
            dlog.warning(
                f"SLOW STEP step={batch_idx} took {step_elapsed:.2f}s | "
                f"seq_len={seq_len} | gpu_alloc={alloc:.0f}MB"
            )

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, image_sizes, answers = batch
        self.model.eval()
        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       pixel_values=pixel_values, image_sizes=image_sizes, max_new_tokens=MAX_LENGTH)
        # turn them back into text, chopping of the prompt
        # important: we don't skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores), sync_dist=True)

        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def on_train_epoch_start(self):
        dlog = self._get_dlog()
        # Reclaim fragmented CUDA memory between epochs to prevent OOM
        torch.cuda.empty_cache()
        alloc, reserved, free = get_gpu_mem_mb()
        dlog.info(f">>> EPOCH {self.current_epoch} START | gpu_alloc={alloc:.0f}MB | gpu_free={free:.0f}MB")
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        dlog = self._get_dlog()
        epoch_elapsed = time.time() - self._epoch_start_time if hasattr(self, '_epoch_start_time') else -1
        alloc, reserved, free = get_gpu_mem_mb()
        dlog.info(
            f"<<< EPOCH {self.current_epoch} END | epoch_time={epoch_elapsed:.1f}s "
            f"| gpu_alloc={alloc:.0f}MB | gpu_free={free:.0f}MB"
        )

    def train_dataloader(self):
        dlog = self._get_dlog()
        dlog.info(f"Creating train DataLoader (dataset_size={len(train_dataset)}, batch_size={self.batch_size}, num_workers=4)")
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4)

config = {"max_epochs": EPOCH,
          # "val_check_interval": 0.2, # how many times we want to validate during an epoch
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 6,
          "lr": 2e-5,
          "batch_size": 1,
          # "seed":2022,
          "num_nodes": 1,
          "warmup_steps": 50,
          "result_path": "/home/chenkuiyun/MLLM/output",
          "verbose": True,
}

model_module = LlavaModelPLModule(config, processor, model)

#Define callbacks
api = HfApi()

class PushToHubCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        pl_module.model.push_to_hub(REPO_ID,
                                    commit_message=f"Training in progress, epoch {trainer.current_epoch}")

    def on_train_end(self, trainer, pl_module):
        print(f"Pushing model to the hub after training")
        pl_module.processor.push_to_hub(REPO_ID,
                                    commit_message=f"Training done")
        pl_module.model.push_to_hub(REPO_ID,
                                    commit_message=f"Training done")

class SaveToDiskCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        dlog = logging.getLogger("debug_train")
        dlog.info(f"[CALLBACK] on_train_epoch_end ENTER | rank={trainer.global_rank} | epoch={trainer.current_epoch}")
        if trainer.global_rank == 0:
            t0 = time.time()
            print(f"Saving model to disk, epoch {trainer.current_epoch}")
            dlog.info(f"[CALLBACK] rank 0 saving model to {SAVE_DIR} ...")
            pl_module.model.save_pretrained(SAVE_DIR)
            pl_module.processor.save_pretrained(SAVE_DIR)
            save_elapsed = time.time() - t0
            dlog.info(f"[CALLBACK] rank 0 save completed in {save_elapsed:.2f}s")
        # Synchronize all ranks — wait for rank 0 to finish saving
        dlog.info(f"[CALLBACK] rank={trainer.global_rank} entering barrier ...")
        t_barrier = time.time()
        trainer.strategy.barrier()
        barrier_elapsed = time.time() - t_barrier
        dlog.info(f"[CALLBACK] rank={trainer.global_rank} barrier passed in {barrier_elapsed:.2f}s")
        dlog.info(f"[CALLBACK] on_train_epoch_end EXIT | rank={trainer.global_rank}")

    def on_train_end(self, trainer, pl_module):
        dlog = logging.getLogger("debug_train")
        dlog.info(f"[CALLBACK] on_train_end ENTER | rank={trainer.global_rank}")
        if trainer.global_rank == 0:
            t0 = time.time()
            print(f"Saving model to disk after training")
            dlog.info(f"[CALLBACK] rank 0 final save to {SAVE_DIR} ...")
            pl_module.model.save_pretrained(SAVE_DIR)
            pl_module.processor.save_pretrained(SAVE_DIR)
            save_elapsed = time.time() - t0
            dlog.info(f"[CALLBACK] rank 0 final save completed in {save_elapsed:.2f}s")
        dlog.info(f"[CALLBACK] rank={trainer.global_rank} entering final barrier ...")
        t_barrier = time.time()
        trainer.strategy.barrier()
        barrier_elapsed = time.time() - t_barrier
        dlog.info(f"[CALLBACK] rank={trainer.global_rank} final barrier passed in {barrier_elapsed:.2f}s")
        dlog.info(f"[CALLBACK] on_train_end EXIT | rank={trainer.global_rank}")



early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")

checkpoint_callback = ModelCheckpoint(
    dirpath='/home/chenkuiyun/MLLM/output/',  # 指定保存路径
    filename='llava-v1.6-mistral-7b-lora-test',  # 指定文件名格式
    save_top_k=1,
    verbose=True,
    #monitor='val_loss',
    mode='min'
)
#Train!
#wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)

trainer = L.Trainer(
        accelerator="gpu",
        devices=4,
        strategy=DeepSpeedStrategy(
            stage=2,
            timeout=timedelta(minutes=30),
        ),
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision="16-mixed",
        log_every_n_steps=10,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        #strategy="ddp_find_unused_parameters_true",
        #logger=wandb_logger,
        #callbacks=[PushToHubCallback(), early_stop_callback],
        callbacks=[SaveToDiskCallback()]
)

trainer.fit(model_module)
