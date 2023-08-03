# coding=utf-8
import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from trl import SFTTrainer

@dataclass
class ScriptArguments:
    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    per_device_train_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")
    dataset_name: Optional[str] = field(default="timdettmers/openassistant-guanaco")
    use_4bit: Optional[bool] = field(default=True)
    use_nested_quant: Optional[bool] = field(default=False)
    bnb_4bit_compute_dtype: Optional[str] = field(default="float16")
    bnb_4bit_quant_type: Optional[str] = field(default="nf4")
    num_train_epochs: Optional[int] = field(default=1)
    fp16: Optional[bool] = field(default=False)
    bf16: Optional[bool] = field(default=False)
    packing: Optional[bool] = field(default=False)
    gradient_checkpointing: Optional[bool] = field(default=True)
    optim: Optional[str] = field(default="paged_adamw_32bit")
    lr_scheduler_type: str = field(default="constant")
    max_steps: int = field(default=10000)
    warmup_ratio: float = field(default=0.03)
    group_by_length: bool = field(default=True)
    save_steps: int = field(default=10)
    logging_steps: int = field(default=10)
    merge_and_push: Optional[bool] = field(default=False)
    output_dir: str = field(default="./results")

def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )
    # Load the entire model onto GPU 0
    # Switch to device_map = "auto" for multi-GPU configurations
    device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        quantization_config=bnb_config, 
        device_map=device_map, 
        use_auth_token=True
    )
    model.config.pretraining_tp = 1 
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM", 
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, peft_config, tokenizer

def train_model(args):
    model, peft_config, tokenizer = create_and_prepare_model(args)
    model.config.use_cache = False
    dataset = load_dataset(args.dataset_name, split="train")
    # Fix the unusual overflow issue in fp16 training.
    tokenizer.padding_side = "right"
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=args.packing,
    )
    trainer.train()
    if args.merge_and_push:
        output_dir = os.path.join(args.output_dir, "final_checkpoints")
        trainer.model.save_pretrained(output_dir)
        # Free up memory for merging weights
        del model
        torch.cuda.empty_cache()
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()
        output_merged_dir = os.path.join(args.output_dir, "final_merged_checkpoint")
        model.save_pretrained(output_merged_dir, safe_serialization=True)

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    train_model(script_args)
