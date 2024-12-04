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
# import dill as pickle
import pickle
from outlines.models.transformers import Transformers
from .infer_server import run_server
import requests
import time
from multiprocessing import Process
from outlines.fsm.json_schema import build_regex_from_schema, get_schema_from_signature
from pydantic import BaseModel

def check_and_start_server(port=5500):
    server_url = f'http://127.0.0.1:{port}/ping'
    
    try:
        response = requests.get(server_url)
        if response.status_code == 200:
            print('Server is already running.')
            return
    except requests.ConnectionError:
        print('Server is not running. Starting a new server...')

    server_process = Process(target=run_server, args=(port,))
    server_process.start()

    # Wait a moment for the server to start
    time.sleep(10)

    try:
        response = requests.get(server_url)
        if response.status_code == 200:
            print('Server started successfully.')
    except requests.ConnectionError:
        print('Failed to start the server.')
        try:
            time.sleep(5)
            response = requests.get(server_url)
        except requests.ConnectionError:
            print('Failed to start the server.')
    
    return server_process

def shutdown_server(server_process):
    if server_process is not None:
        try:
            server_process.terminate()
            server_process.join()
            server_process = None
        except requests.ConnectionError:
            print('Failed to shut down the server.')



def save_to_pkl(file_path, data):
    with open(file_path+"_backup.pkl", 'wb') as f:
        pickle.dump(data, f)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    os.remove(file_path+"_backup.pkl")


def format_infer(parms):
    output_formater=parms.get("output_formater", None)
    model_folder=parms.get("model_folder", None)
    model_inputs_lst=parms.get("model_inputs_lst", None)
    max_new_tokens=parms.get("max_new_tokens", None)
    max_length=parms.get("max_length", None)
    save_step=parms.get("save_step", 5)
    intermediates_progress_file=parms["intermediates_progress_file"]
    bar_position=parms.get("p_idx", 0)

    # model_folder, model_inputs_lst, max_new_tokens, max_length
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        )
    model=AutoModelForCausalLM.from_pretrained(model_folder, quantization_config=bnb_config, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_folder)

    format_model=Transformers(model=model, tokenizer=tokenizer)
    generator = outlines.generate.json(format_model, output_formater)

    rng = torch.Generator(device="cuda")
    rng.manual_seed(789001)

    ans=[]
    with tqdm(total=len(model_inputs_lst), position=bar_position+1, desc=f"Task ID: {bar_position}") as pbar:
        for i in range(len(model_inputs_lst)):
            start_idx=model_inputs_lst[i][0]
            query=model_inputs_lst[i][1]

            character = generator(query, rng=rng)
            decoded=repr(character)
            
            if (i +1) % save_step==0:
                save_to_pkl(intermediates_progress_file, ans)
            
            ans.append((start_idx, decoded[0], query))
            pbar.update(1)

    save_to_pkl(intermediates_progress_file, ans)

    return ans


def single_infer(parms):
    model_folder=parms.get("model_folder", None)
    model_inputs_lst=parms.get("model_inputs_lst", None)
    max_new_tokens=parms.get("max_new_tokens", None)
    max_length=parms.get("max_length", None)
    save_step=parms.get("save_step", 5)
    intermediates_progress_file=parms["intermediates_progress_file"]
    bar_position=parms.get("p_idx", 0)

    # model_folder, model_inputs_lst, max_new_tokens, max_length
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        )
    model=AutoModelForCausalLM.from_pretrained(model_folder, quantization_config=bnb_config, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_folder)

    ans=[]
    with tqdm(total=len(model_inputs_lst), position=bar_position+1, desc=f"Task ID: {bar_position}") as pbar:
        for i in range(len(model_inputs_lst)):
            start_idx=model_inputs_lst[i][0]
            query=model_inputs_lst[i][1]
            encodeds = tokenizer(query, return_tensors="pt", add_special_tokens=False)
            model_inputs = encodeds.to("cuda:0")
            if max_new_tokens is None:
                generated_ids = model.generate(
                            **model_inputs, 
                            max_length=max_length, 
                            do_sample=False, 
                            pad_token_id=tokenizer.eos_token_id
                            )
            else:
                generated_ids = model.generate(
                            **model_inputs, 
                            max_new_tokens=max_new_tokens, 
                            do_sample=False, 
                            pad_token_id=tokenizer.eos_token_id
                            )
            
            if (i +1) % save_step==0:
                save_to_pkl(intermediates_progress_file, ans)
            
            decoded = tokenizer.batch_decode(generated_ids)
            ans.append((start_idx, decoded[0], query))
            pbar.update(1)
    
    save_to_pkl(intermediates_progress_file, ans)

    return ans

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
    return regex_str

class InferAgent:
    def initialize(self, load_model=False):
        if self.model is not None:
            return
        
        self.device = "cuda:0"

        bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        # bnb_8bit_use_double_quant=True,
        # bnb_8bit_quant_type="nf4",
        # bnb_8bit_compute_dtype=torch.bfloat16
        )
        
        with open(self.meta_model_dic["plain_query_file"], "r") as f:
                self._plain_query_template = "".join(f.readlines())

        self._few_shot_query_template=None
        if os.path.exists(self.meta_model_dic["query_file"]):
            with open(self.meta_model_dic["query_file"], "r") as f:
                self._few_shot_query_template="".join(f.readlines()) 

        if load_model:
            self.model = AutoModelForCausalLM.from_pretrained(self.meta_model_dic["model_folder"], quantization_config=bnb_config, device_map=self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.meta_model_dic["model_folder"])

    def new_infer(self, question, input_content, output_fd=None, verbose=False, max_new_tokens=1000, max_length=None, few_shot=True, re_try=0, formatter=None):
        self.initialize()
        port=5501
        server_process=check_and_start_server(port)
        _model_params={"max_length":max_length,
            "max_new_tokens":max_new_tokens, 
            "do_sample":True}
        model_params={k:v for k,v in _model_params.items() if v is not None}

        parms={"model_folder":self.meta_model_dic["model_folder"]}
        parm_pkl=pjoin(self.meta_model_dic["model_folder"], 'parms.pkl')
        with open(parm_pkl, 'wb') as file:
            pickle.dump(parms, file)

        regex_str=get_regex_str(formatter)
        formatter_pkl=pjoin(self.meta_model_dic["model_folder"], 'formatter.pkl')
        with open(formatter_pkl, 'wb') as file:
            pickle.dump(regex_str, file)

        response = requests.get(f'http://127.0.0.1:{port}/connect', params = {"path": parm_pkl, "formatter_path":formatter_pkl})
        if response.status_code!= 200:
            return response.json()


        query_template=self._plain_query_template if not few_shot or self._few_shot_query_template is None else self._few_shot_query_template
        query_text=query_template.format(infer_question=question, infer_input=input_content)
        query_pkl=pjoin(self.meta_model_dic["model_folder"], "query.pkl")
        with open(query_pkl, 'wb') as file:
            pickle.dump({"query_text": query_text, "parms":model_params}, file)

        response = requests.get(f'http://127.0.0.1:{port}/infer', params = {"path": query_pkl})
        while True:
            if input("")=="q":
                break
        shutdown_server(server_process)
        return response.json()

    def infer(self, question, input_content, output_fd=None, verbose=False, max_new_tokens=1000, max_length=None, few_shot=True, re_try=0):
        self.initialize(load_model=True)

        if few_shot and self._few_shot_query_template is None:
            print("No few shot example given, switch to plain query")

        query_template=self._plain_query_template if not few_shot or self._few_shot_query_template is None else self._few_shot_query_template

        query_text=query_template.format(infer_question=question, infer_input=input_content)
        if verbose==2:
            print(query_text)
        encodeds = self.tokenizer(query_text, return_tensors="pt", add_special_tokens=False)
        model_inputs = encodeds.to(self.device)
        if verbose:
            print("Infering")
        
        cleaned_result_dic={}
        try:

            _parms={
            "max_length":max_length,
            "max_new_tokens":max_new_tokens, 
            "do_sample":True,
            "pad_token_id":self.tokenizer.eos_token_id}
            _parms={k:v for k,v in _parms.items() if v is not None}

            generated_ids = self.model.generate(
                        **model_inputs, **_parms)
            _decoded = self.tokenizer.batch_decode(generated_ids)
            _cleaned_result=self.clean_output(_decoded[0], query_text)
            cleaned_result_dic["generated_text"]=_cleaned_result

            for i in range(re_try):
                generated_ids = self.model.generate(
                    **model_inputs, **_parms)
                _decoded = self.tokenizer.batch_decode(generated_ids)
                _cleaned_result=self.clean_output(_decoded[0], query_text)
                cleaned_result_dic[f"generated_text_{i+1}"]=_cleaned_result

        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("CUDA Full, please ask question again.", e)
            assert False, "CUDA Full, please ask question again."
        if verbose==1:
            print("Generated Text: ", cleaned_result_dic["generated_text"])
        if verbose==2:
            # print("Generated Text: ", decoded[0])    
            print("Cleaned Text: ", cleaned_result_dic)

        if output_fd is not None:
            with open(pjoin(output_fd, f"answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
                    json.dump({"input_content":input_content,"generated_text": json.dumps(cleaned_result_dic)}, f, indent=4)
        else:
            return cleaned_result_dic
        
    @staticmethod
    def is_gpu_memory_available():
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], capture_output=True, text=True)
            gpu_memory = result.stdout.strip().split('\n')[0].split(',')
            used, total = map(int, gpu_memory)
            print("GRAM: ",used, total)
            return min(2, int(int(total)/int(used))-1)
        except Exception as e:
            print(f"Error obtaining GPU memory usage: {e}")
            return 1

    def step_prep(self, infer_spl_lst, few_shot=True):
        if few_shot and self._few_shot_query_template is None:
            print("No few shot example given, switch to plain query")
        
        query_template=self._plain_query_template if not few_shot or self._few_shot_query_template is None else self._few_shot_query_template

        step_res=[]
        for spl in infer_spl_lst:
            query_text=query_template.format(infer_question=spl[1]["question"], infer_input=spl[1]["input"])
            step_res.append((sql[0],query_text))
        return step_res

    @staticmethod
    def load_prgress_result(folder_path, infer_spl_lst=None):
        concatenated_list = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if filename.endswith('.pkl') and os.path.getsize(file_path)>1:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    concatenated_list.extend(data)

        subprocess.run(["rm", "-rf", folder_path])
        os.makedirs(folder_path, exist_ok=True)
        print("intermediates_folder created at: ", folder_path)
        with open(folder_path+"/progress.pkl", 'wb') as f:
            pickle.dump(concatenated_list, f)

        if infer_spl_lst is None:
            concatenated_list=sorted(concatenated_list, key=lambda x: x[0])
            return [k[1] for k in concatenated_list]
        
        indices_to_exclude = {x[0] for x in concatenated_list}
        new_infer_spl_lst=[(i, x) for i, x in enumerate(infer_spl_lst) if i not in indices_to_exclude]
        assert len(new_infer_spl_lst)>0, "Inferring task finished, no new task to do"
        return new_infer_spl_lst

    def batch_infer(self, infer_samples, question=None, output_fd=None, verbose=False, max_new_tokens=1000, max_length=None, few_shot=True, parallel=False, save_step=5, resume=True, re_try=0, formatter=None):
        infer_spl_lst=self.dataset_creator.create_dataset(infer_samples, question=question)

        intermediates_folder=pjoin(self.meta_model_dic["model_folder"], "infer_intermediates")

        res=[]
        if not parallel:
            if resume and os.path.exists(intermediates_folder):
                infer_spl_lst=InferAgent.load_prgress_result(intermediates_folder, infer_spl_lst)
                infer_spl_lst=[x[1] for x in infer_spl_lst]
            else:
                os.makedirs(intermediates_folder, exist_ok=True)
            for i in tqdm(range(len(infer_spl_lst))):
                spl=infer_spl_lst[i]
                ans=self.infer(spl["question"], spl["input"], output_fd=None, verbose=verbose, max_new_tokens=max_new_tokens, max_length=max_length, few_shot=few_shot, re_try=re_try)
                res.append(ans)
                
                if (i +1) % save_step==0:
                    save_to_pkl(os.path.join(intermediates_folder, f"progress.pkl"), res)

        
        if parallel:
            self.initialize()
            if type(parallel)==int:
                parallel_task_num=parallel
            else:
                parallel_task_num=InferAgent.is_gpu_memory_available()

            if parallel_task_num==0:
                assert False, "No GPU RAM available, please closing other GPU processes"


            if resume and os.path.exists(intermediates_folder):
                infer_spl_lst=InferAgent.load_prgress_result(intermediates_folder, infer_spl_lst)
            else:
                os.makedirs(intermediates_folder, exist_ok=True)
                
            print("parallel_task_num:", parallel_task_num)

            task_args=[]
            split_num=len(infer_spl_lst)//parallel_task_num+1
            for gi in range(parallel_task_num):
                start_idx = split_num * gi
                end_idx = start_idx + split_num
                sql_lst=infer_spl_lst[start_idx:end_idx]
                print(f"Task {gi}", len(sql_lst))
                if len(sql_lst)>0:
                    step_res= self.step_prep(sql_lst, few_shot)
                    task_args.append({
                        "model_folder":self.meta_model_dic["model_folder"], 
                        "model_inputs_lst":step_res,
                        "max_new_tokens":max_new_tokens,
                        "max_length":max_length,
                        "save_step":save_step,
                        "intermediates_progress_file":f"{intermediates_folder}/progress_gid_{gi}.pkl",
                        "p_idx":gi,
                        "start_idx":start_idx,
                        "output_formater":formatter
                    })
                         
            if formatter is None:
                with multiprocessing.Pool(processes=len(task_args)) as pool:
                    progress_ct = pool.map(single_infer, task_args)    
            else:
                with multiprocessing.Pool(processes=len(task_args)) as pool:
                    progress_ct = pool.map(format_infer, task_args)    

            # with ProcessPoolExecutor() as executor:
            #     progress_ct = list(executor.map(single_infer, task_args))

            tkn_res=InferAgent.load_prgress_result(intermediates_folder)
            
            cleaned_result=[]
            for tkns in tkn_res:
                cleaned_result.append({"generated_text":self.clean_output(tkns[0], tkns[1])})
                
            res.extend(cleaned_result)

        if isinstance(infer_samples, pd.DataFrame):
            res_df =infer_samples.copy().reset_index(drop=True)
            res_df=pd.concat([res_df, pd.DataFrame(res)], axis=1)
        else:
            res_df = pd.DataFrame({"question":[s["question"] for s in infer_spl_lst], "input":[s["input"] for s in infer_spl_lst]})
            res_df=pd.concat([res_df, pd.DataFrame(res)], axis=1)

        if output_fd is not None:
            res_df.to_csv(pjoin(output_fd, f"batch_infer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)
        else:
            return res_df
        
    def folder_infer(self, folder_path, question=None, output_file=None, verbose=False, max_new_tokens=1000, max_length=None, few_shot=True, formatter=None):
        all_dfs=InferAgent.concat_files_in_folder(folder_path, iterator=True)
        for filename, inf_df in all_dfs:
            print("Infering file: ",filename)
            if formatter is not None:
                inf_df=formatter(inf_df)
            ans=self.batch_infer(inf_df, question=question, verbose=verbose, max_new_tokens=max_new_tokens, max_length=max_length, few_shot=few_shot, resume=False)

            if output_file:
                InferAgent.save_or_update_sheets(output_file, filename.split(".")[0], ans)
    
    @staticmethod
    def concat_files_in_folder(folder_path, iterator=False):
        dfs = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
                    
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            if iterator:
                yield filename, df
            else:
                dfs.append(df)
        
        # Concatenate all dataframes in the list
        if not iterator:
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        
    @staticmethod
    def save_or_update_sheets(file_path, df_key, _df):
        if not os.path.exists(file_path):
            pd.DataFrame().to_excel(file_path, index=False)
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            _df.to_excel(writer, sheet_name=df_key, index=False)
