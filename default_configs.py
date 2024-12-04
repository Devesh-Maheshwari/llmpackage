import pandas as pd
import random
import re
import json
import datetime
import os
import ast
import shutil
import transformers
import torch
import peft
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from os.path import join as pjoin


class DefaultConfig:
    def __init__(self, task_folder) -> None:
        self.task_folder=task_folder
    
    def get_default_config(self):
        default_configs={}
        default_configs["dafault_peft_config"] = peft.LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=10,
                bias="none",
                task_type="CAUSAL_LM"
            )
        
        
        default_configs["default_nf4_config_dict"]={"load_in_4bit":True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": torch.bfloat16
            }
        
        finetuned_model_path = pjoin(self.task_folder, "finetuned_model")
        os.makedirs(finetuned_model_path, exist_ok=True)
        default_configs["deafult_tf_train_args"] = transformers.TrainingArguments(
            output_dir=finetuned_model_path,
            max_steps=50,
            per_device_train_batch_size=8,
            # warmup_steps=5,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="steps",
            eval_steps=10,
            learning_rate=3e-4,
            bf16=True,
            lr_scheduler_type='polynomial',
            lr_scheduler_kwargs={'power':1.5, 'lr_end':1e-5}
            )
        
        default_configs["default_sft_config"]={
                "max_seq_length": 128,
                "train_end": 500,
                "eval_begin": 650,
            }
        
        return default_configs