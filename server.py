import json
import llmpackage_hilabs
import pandas as pd
import requests
import time
from multiprocessing import Process
from outlines.fsm.json_schema import build_regex_from_schema, get_schema_from_signature
from pydantic import BaseModel
import threading
from llmpackage_hilabs.infer_server import run_server
from functools import partial

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
import argparse

llmpackage_hilabs.config_properties.root_task_folder="/home/ec2-user/ccda_summarize/temp"
llmpackage_hilabs.config_properties.model_folder_dic={
    "llama":"/home/ec2-user/ccda_summarize/cdi-llm-ccda-summarization/model/llama/Meta-Llama-3-8B-Instruct",
    "llama--refueled":"/home/ec2-user/ccda_summarize/cdi-llm-ccda-summarization/model/Llama-3-Refueled",
    "mistral":"/home/ec2-user/ccda_summarize/cdi-llm-ccda-summarization/model/mistral_instruct/Mistral-7B-Instruct-v0.2",
    "mistral--3":"/home/ec2-user/ccda_summarize/cdi-llm-ccda-summarization/model/mistral_instruct/Mistral-7B-v0.3"}
    
from enum import Enum
from typing import List, Dict

from pydantic import BaseModel, constr, conlist
from outlines.models.transformers import Transformers
from typing_extensions import TypedDict

import outlines
import torch

class UMLS_Category(TypedDict):  # Inherits from BaseModel instead of dict
    entity: str
    semantic_type: str

class Procedure(BaseModel):  # Inherits from BaseModel instead of dict
    procedure_entity: str
    UMLS_Category: str

class Diesease(BaseModel):  # Inherits from BaseModel instead of dict
    disease_name: str

class Medication(BaseModel):  # Inherits from BaseModel instead of dict
    medication_name: str
    dosage: str
    route: str
    brand_name: str

class Test(BaseModel):
    test_name: str

class Clinical_Info(BaseModel):
    procedures: conlist(Procedure)
    medications: conlist(Medication)
    diseases: conlist(Diesease, min_length=1)
    clinical_test: conlist(Test) 


 

ques="Extract the a Clinical_Info (Procedures, Medications, Diseases and Test) from a following clinical summary paragraph:"
df=pd.DataFrame({"input":["The patient was suffering severe agitation by blood tests and ultrasound. Patient was administered Versed IM 2 mg on 07/24/99.", 
                        "On 07/21/2004, the patient was diagnosed with tonsillitis and was given Cloxapen oral tablet for 10 days .",
                        "The patient had a spinal cord stimulator trial on 07/21/99 , the patient had an epidural trial on 07/24/99.",
                        "Aortic valve replacement in 11/96 with a St. Jude valve and chronic Coumadin therapy ; breast biopsy negative in 1984 ; appendectomy age 15 ; tonsillectomy and adenoidectomy age 6 ; hysterotomy age 30 ; C-section age 37 due to placenta previa ."
                        ], "answer": [json.dumps({"procedures":[{'procedure_entity': 'ultrasound', 'UMLS_Category': 'Diagnostic Procedure'}],
                        "medications": [{'medication_name': 'Midazolam', 'dosage': '2 mg', "route": 'IM', "brand_name": "Versed"}],
                        "diseases":["agitation"],
                        "clinical_test":["blood tests"]}), 
            json.dumps({"procedures":[],
                        "medications": [{'medication_name': 'Cloxacillin', 'dosage': 'not mentioned', "route": 'oral', "brand_name": "Cloxapen"}],
                        "diseases":["tonsillitis"],
                        "clinical_test":[]}), 
            json.dumps({"procedures":[{'procedure_entity': 'spinal cord stimulator', 'UMLS_Category': 'Therapeutic or Preventive Procedure'},
                                    {'procedure_entity': 'epidural trial', 'UMLS_Category': 'Therapeutic or Preventive Procedure'}],
                        "medications": [],
                        "diseases":[],
                        "clinical_test":[]}), 
            json.dumps({"procedures":[{'procedure_entity': 'aortic valve replacement', 'UMLS_Category': 'Therapeutic or Preventive Procedure'},
                                    {'procedure_entity': 'tonsillectomy', 'UMLS_Category': 'Therapeutic or Preventive Procedure'},
                                    {'procedure_entity': 'adenoidectomy', 'UMLS_Category': 'Therapeutic or Preventive Procedure'},
                                    {'procedure_entity': 'C-section', 'UMLS_Category': 'Therapeutic or Preventive Procedure'},],
                        "medications": [{'medication_name': 'warfarin', 'dosage': 'not mentioned', "route": 'not mentioned', "brand_name": "Coumadin"}],
                        "diseases":["placenta previa"],
                        "clinical_test":["breast biopsy"]})
    ]
})   

def check_and_start_server(host="127.0.0.1",port=5500):
    server_url = f'http://{host}:{port}/ping'
    try:
        response = requests.get(server_url, timeout=3)
        if response.status_code == 200:
            print('Server is already running.')
            return
    except requests.ConnectionError:
        print('Server is not running. Starting a new server...')

    print("Starting the server...", server_url)
    server_process = Process(target=run_server, args=(host, port))
    server_process.start()
    
    time.sleep(10)
    try:
        response = requests.get(server_url)
        if response.status_code == 200:
            print('Server started successfully.')
    except requests.ConnectionError:
        print('Server not started. Retrying...')
        try:
            time.sleep(5)
            response = requests.get(server_url)
        except requests.ConnectionError:
            print('Failed to start the server.')
    
    print('Server is running.', server_process)
    return server_process


def shutdown_server(server_process):
    assert server_process is not None, "No server process provided to shut down."
    try:
        server_process.terminate()
        server_process.join()
        server_process = None
    except requests.ConnectionError:
        print('Failed to shut down the server.')


def get_regex_str(schema_object):
    if isinstance(schema_object, type(BaseModel)):
        schema = json.dumps(schema_object.model_json_schema())
        regex_str = build_regex_from_schema(schema)
    elif callable(schema_object):
        schema = json.dumps(get_schema_from_signature(schema_object))
        regex_str = build_regex_from_schema(schema)
    elif isinstance(schema_object, str):
        schema = schema_object
        regex_str = build_regex_from_schema(schema)
    else:
        regex_str=None
    return regex_str


def initialize_server(host, port, cfg_parms, formatter_re):
    server_process=check_and_start_server(host=host, port=port)
    print("Server Inited")

    cfg_pkl_path=pjoin(cfg_parms["model_folder"], 'config_.pkl')
    with open(cfg_pkl_path, 'wb') as file:
        pickle.dump(cfg_parms, file)

    regex_str=formatter_re
    ot_formatter_pkl_path=pjoin(cfg_parms["model_folder"], 'formatter.pkl')
    with open(ot_formatter_pkl_path, 'wb') as file:
        pickle.dump(regex_str, file)

    response = requests.get(f'http://{host}:{port}/connect', params = {"path": cfg_pkl_path, "formatter_path":ot_formatter_pkl_path})
    if response.status_code!= 200:
        return response.json()

    return server_process
    

def start_server(model_folder, host, port):
    print(f"Starting server at {host}, {port}")
    tm=llmpackage.BaseModel("mistral--3",
                            "mistral--3_clinical_info_4_attr", 
                            examples=df,
                            load_task=False, 
                            question=ques)
    tm.create_model()
    tm.initialize()

    _cfg_parms_dic={
        "model_folder":tm.meta_model_dic["model_folder"],
        "query_template":tm._few_shot_query_template
        }

    server_process=initialize_server(host, port, _cfg_parms_dic, get_regex_str(Clinical_Info))
    
    return server_process


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start a server.')
    
    # Named arguments with defaults
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host address where the server will run (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5501,
                        help='Port number where the server will listen (default: 5001)')
    
    # Positional arguments that are optional
    parser.add_argument('model_folder', nargs='?', help=argparse.SUPPRESS)
    parser.add_argument('positional_host', nargs='?', help=argparse.SUPPRESS)
    parser.add_argument('positional_port', nargs='?', type=int, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Determine host and port based on provided arguments
    model_folder = args.model_folder
    host = args.positional_host if args.positional_host else args.host
    port = args.positional_port if args.positional_port else args.port

    server_process = start_server(model_folder, host, port)
    print(f"{server_process} running...")

    while True:
        input_content = input("CLOSE y/n: ")
        if input_content.lower() == "y":
            shutdown_server(server_process)
            break