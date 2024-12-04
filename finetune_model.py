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
from .task_management import TaskManager
from .infer_agent import InferAgent
from .default_configs import DefaultConfig
from functools import partial
from datasets import Dataset

class FinetuneModel(TaskManager, InferAgent):
    def create_model(self, finetuned_model_folder):
        tkn={
                "model_name": self.model_name,
                "model_type": self.model_type,
                "base_model_name": self.base_model_name,
                "finetuned_model": "True",
                "finetuned_time": datetime.datetime.now().strftime('%m%d%Y_%H%M$S'),
                "model_folder": finetuned_model_folder,
                "query_file": pjoin(self.task_folder, "data", "few_shot_query.txt"),
                "plain_query_file": pjoin(self.task_folder, "data", "plain_query.txt")
            }
        
        
        with open(pjoin(self.task_folder, "mdl.tkn"), "w") as f:
            json.dump(tkn, f, indent=4)
        
        self.meta_model_dic=tkn
    
    def load_model(self):
        with open(pjoin(self.task_folder, "mdl.tkn"), "r") as f:
            self.meta_model_dic=json.load(f)
        
    def prepare_training_data(self, training_samples=None, question=None, few_shot=True):
        assert training_samples is not None, "Training samples are required to create a model"

        _query_path=pjoin(self.task_folder, "data", "few_shot_finetune_query.txt") if few_shot else pjoin(self.task_folder, "data", "plain_finetune_query.txt")
        with open(_query_path, "r") as f:
            self._finetune_query_template = "".join(f.readlines())

        
        training_spl_lst=self.dataset_creator.create_dataset(training_samples, question=question)
        print("Training data nums: ", len(training_spl_lst))
        with open (pjoin(self.task_folder, "finetune_dataset.json"), "w") as f:
            json.dump(training_spl_lst, f, indent=4)
        self.flags["training_data_created"]=True
        print("Training data ready")


       
    def finetune_model(self, training_samples=None, question=None, few_shot=True, peft_config=None, nf4_config=None, tf_train_args=None, sft_config=None):
        if not self.flags.get("training_data_created", False):
            self.prepare_training_data(training_samples=training_samples, question=question, few_shot=True)


        finetune_configs={}
        default_configs=DefaultConfig(self.task_folder).get_default_config()

        _peft_config=default_configs["dafault_peft_config"] if peft_config is None else peft_config
        finetune_configs["LoraConfig"]={key: str(val) for key, val in re.findall(r'(\w+)\s*=\s*(<[^>]*>|{[^}]*}|[^\s,]+|".*?")', str(_peft_config))}


        nf4_config_dict=default_configs["default_nf4_config_dict"] if nf4_config is None else nf4_config
        _nf4_config = transformers.BitsAndBytesConfig.from_dict(nf4_config_dict)
        finetune_configs["nf4_config"]={key: str(val) for key, val in nf4_config_dict.items()}

        _tf_train_args=default_configs["deafult_tf_train_args"] if tf_train_args is None else tf_train_args
        finetune_configs["TrainingArguments"]=_tf_train_args.to_dict()

        _sft_config=default_configs["default_sft_config"] if sft_config is None else sft_config
        finetune_configs["sft_config"]=_sft_config


        base_model = AutoModelForCausalLM.from_pretrained(
                self.model_folder_dic[self.base_model_name],
                device_map='cuda:0',
                quantization_config=_nf4_config,
                use_cache=False
            )
        tokenizer = AutoTokenizer.from_pretrained(self.model_folder_dic[self.base_model_name])
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.pad_token else tokenizer.eos_token
        print(tokenizer.pad_token)
        tokenizer.padding_side = "left"
        try:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        except:
            tokenizer.pad_token = tokenizer.unk_token if tokenizer.pad_token else tokenizer.eos_token
        model = peft.prepare_model_for_kbit_training(base_model)
        model = peft.get_peft_model(model, _peft_config)
    

        def sft_formater(sample, query_template=None):
            # return f"Question: Given rate a random number. \n Input: Range: 1 - 100.\n {str(random.randint(1,100))}"
            if query_template is None:
                return f"Question: Given rate a random number. \n Input: Range: 1 - 100.\n {str(random.randint(1,100))}"
            query=query_template.format(infer_question=str(sample["question"]), 
                                        infer_input=str(sample["input"]), 
                                        infer_answer=str(sample["answer"]))
            return query
    
        create_single_data = lambda sample: sft_formater(sample, query_template=self._finetune_query_template)

        with open (pjoin(self.task_folder, "finetune_dataset.json"), "r") as f:
            training_spl_lst=json.load(f)
        
        dsdf=pd.DataFrame({"text": [sft_formater(x, query_template=self._finetune_query_template) for x in training_spl_lst]})
        print(len(dsdf["text"].iloc[0]))
        trainer = SFTTrainer(
            model=model,
            peft_config=_peft_config,
            max_seq_length=_sft_config["max_seq_length"],
            # formatting_func=create_single_data,
            dataset_text_field='text',
            tokenizer=tokenizer,
            packing=True,
            args=_tf_train_args,
            train_dataset=Dataset.from_pandas(dsdf),
            eval_dataset=Dataset.from_pandas(dsdf.loc[:10]),
            )

        trainer.train()
        user_decision=input("Do you want to save the finetuned model? (y/n)")
        if user_decision.lower()=="y":
            finetuned_model_folder=pjoin(self.task_folder, "model")
            shutil.rmtree(finetuned_model_folder, ignore_errors=True)
            os.makedirs(finetuned_model_folder)

            trainer.save_model(finetuned_model_folder)
            with open(pjoin(self.task_folder, "finetune_info.json"), "w") as f:
                json.dump(finetune_configs,f, indent=4)
            
            self.create_model(finetuned_model_folder)