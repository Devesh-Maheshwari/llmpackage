import sys
import pandas as pd
import random
import re
import json
import transformers
import torch
import os
import peft
import outlines
import multiprocessing
import subprocess
from itertools import chain
from trl import SFTTrainer
from concurrent.futures import ProcessPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from os.path import join as pjoin
from datetime import datetime
from collections import defaultdict as ddict
from tqdm.auto import tqdm
import pickle
from outlines.models.transformers import Transformers
from flask import Flask, request, jsonify
import threading
import signal
from outlines.samplers import Sampler, multinomial
import time

class InferServer:
    def __init__(self, host='127.0.0.1', port=5000):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.device = "cuda"
        self.server_thread = None
        self.model_in_use=False
        self.model_lock=False

        @self.app.route('/ping', methods=['GET'])
        def ping():
            return jsonify({'message': 'pong'})
        
        @self.app.route('/connect', methods=['GET'])
        def connect():
            pkl_path = request.args.get('path')
            with open(pkl_path, 'rb') as f:
                model_params = pickle.load(f)
            
            pkl_path = request.args.get('formatter_path')
            print(pkl_path)
            with open(pkl_path, 'rb') as f:
                regex_str = pickle.load(f)
            return self.handle_connect(model_params, regex_str)
        
        @self.app.route('/infer', methods=['GET'])
        def infer():
            pkl_path = request.args.get('path')
            with open(pkl_path, 'rb') as f:
                query_info = pickle.load(f)
            query_text = query_info.get('query_text')
            parms = query_info.get('infer_parms')
            return self.handle_infer(query_text, parms)

        @self.app.route('/pid', methods=['GET'])
        def get_pid():
            return jsonify({'pid': os.getpid()})

    def monitor_flag(self):
        while True:
            if self.model_in_use and not self.model_lock:
                print("Model not in use Check 1.")
                time.sleep(30)
                if self.model_in_use and not self.model_lock:
                    print("Model not in use Check 2.")
                    time.sleep(30)
                    if self.model_in_use and not self.model_lock:
                        print("Model not in use Check 3.")
                        del self.model
                        del self.tokenizer # If flag is False, free the memory
                        self.model_in_use = False
                        break
                else:
                    print("Model in use.")
            time.sleep(300)

    def start_server(self):
        threading.Thread(target=self.app.run, kwargs={'host': self.host, 'port': self.port}).start()

    def handle_connect(self, model_parms, regex_str):
        model_folder=model_parms.get("model_folder", None)
        self.query_template=model_parms.get("query_template", "{infer_question}\n {infer_input}")

        _bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        # if bnb_config is None else bnb_config
        self.model=AutoModelForCausalLM.from_pretrained(model_folder, quantization_config=_bnb_config, device_map="cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(model_folder)
        
        self.regex_str=regex_str
        if regex_str is not None:
            self.rng = torch.Generator(device="cuda")
            self.rng.manual_seed(789001)
            format_model=Transformers(model=self.model, tokenizer=self.tokenizer)
            generator = outlines.generate.regex(format_model, regex_str, multinomial())
            generator.format_sequence = lambda x: json.loads(x)
            self.generator = generator

        self.model_in_use=True
        self.timer_thread = threading.Thread(target=self.monitor_flag)  # Set up the monitoring thread
        self.timer_thread.start() 
        print("conneted")
        return jsonify({'message': 'Model loaded.'})

    def handle_infer(self, query, parms):
        self.model_lock=True
        query_text=self.query_template.format(infer_question=query["question"], infer_input=query["input"])
        # return {"t":query_text, "q":query}
        # print(query_text)
        # print("\n\n(())",query)
        
        encodeds = self.tokenizer(query_text, return_tensors="pt", add_special_tokens=False)
        model_inputs = encodeds.to(self.device)
        if self.regex_str is not None:
            infer_results={}
            for retry_idx in range(parms.get("retry", 0)+1):
                character = self.generator(query_text, rng=self.rng)
                infer_results[f"generated_text_{retry_idx}"]=character
        else:
            infer_parms={"max_length":parms.get("max_length", None),
                    "max_new_tokens":parms.get("max_new_tokens", 1000), 
                    "do_sample":parms.get("do_sample", bool(parms.get("retry", False))),
                    "pad_token_id":self.tokenizer.eos_token_id}
            infer_parms={k:v for k,v in infer_parms.items() if v is not None}

            infer_results={}
            for retry in range(parms.get("retry", 0)+1):
                generated_ids = self.model.generate(**model_inputs, **infer_parms)
                decoded = self.tokenizer.batch_decode(generated_ids)
                infer_results[f"generated_text_{retry}"]=decoded[0]

        self.model_lock=False
        return jsonify(infer_results) 

def run_server(host=None,port=None):
    server = InferServer(host=host, port=port)
    server.start_server()