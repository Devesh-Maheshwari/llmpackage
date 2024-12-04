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
import threading
from .server import start_server, initialize_server, get_regex_str, shutdown_server
import copy


port=5501
host='0.0.0.0'


# def check_and_start_server(host="127.0.0.1",port=5500):
#     server_url = f'http://{host}:{port}/ping'
    
#     try:
#         response = requests.get(server_url)
#         if response.status_code == 200:
#             print('Server is already running.')
#             return
#     except requests.ConnectionError:
#         print('Server is not running. Starting a new server...')

#     server_process = Process(target=run_server, args=(host, port))
#     server_process.start()

#     # Wait a moment for the server to start
#     time.sleep(10)

#     try:
#         response = requests.get(server_url)
#         if response.status_code == 200:
#             print('Server started successfully.')
#     except requests.ConnectionError:
#         print('Failed to start the server.')
#         try:
#             time.sleep(5)
#             response = requests.get(server_url)
#         except requests.ConnectionError:
#             print('Failed to start the server.')
    
#     return server_process



def save_to_pkl(file_path, data):
    with open(file_path+"_backup.pkl", 'wb') as f:
        pickle.dump(data, f)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    os.remove(file_path+"_backup.pkl")


def single_process_infer(parms):
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


class InferAgent:
    def initialize(self, load_model=False):
        if self.model is not None:
            return
        
        self.device = "cuda:0"
        self.model_in_use = False
        self.model_lock = False
        
        with open(self.meta_model_dic["plain_query_file"], "r") as f:
                self._plain_query_template = "".join(f.readlines())

        self._few_shot_query_template=None
        if os.path.exists(self.meta_model_dic["query_file"]):
            with open(self.meta_model_dic["query_file"], "r") as f:
                self._few_shot_query_template="".join(f.readlines())

    def load_model(self):
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # bnb_8bit_use_double_quant=True,
        # bnb_8bit_quant_type="nf4",
        # bnb_8bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.meta_model_dic["model_folder"], quantization_config=bnb_config, device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.meta_model_dic["model_folder"])
        print("model_loaded")
        self.model_in_use = True
        self.timer_thread = threading.Thread(target=self.monitor_flag)  # Set up the monitoring thread
        self.timer_thread.start() 

    def monitor_flag(self):
        while True:
            time.sleep(60)
            if self.model_in_use and not self.model_lock:
                del self.model
                del self.tokenizer # If flag is False, free the memory
                self.model_in_use = False
                break
    
    # @staticmethod
    # def init_server(meta_model_dic, formatter):

    #     port=5501
    #     host='0.0.0.0'
    #     server_process=check_and_start_server(host=host, port=port)
        
    #     cfg_parms={
    #         "model_folder":meta_model_dic["model_folder"]
    #         }
    #     cfg_pkl_path=pjoin(meta_model_dic["model_folder"], 'config_.pkl')
    #     with open(cfg_pkl_path, 'wb') as file:
    #         pickle.dump(cfg_parms, file)

    #     regex_str=get_regex_str(formatter)
    #     ot_formatter_pkl_path=pjoin(meta_model_dic["model_folder"], 'formatter.pkl')
    #     with open(ot_formatter_pkl_path, 'wb') as file:
    #         pickle.dump(regex_str, file)

    #     response = requests.get(f'http://{host}:{port}/connect', params = {"path": cfg_pkl_path, "formatter_path":ot_formatter_pkl_path})
    #     if response.status_code!= 200:
    #         return response.json()

    #     return server_process, host, port

    @staticmethod
    def api_infer(meta_model_dic, query_text_lst, verbose=False, model_params=None, formatter=None, save_step=5):
        _cfg_parms_dic={
            "model_folder":meta_model_dic["model_folder"],
            "query_template":meta_model_dic["query_template"]
            }
        server_process=initialize_server(host, port, _cfg_parms_dic, get_regex_str(formatter))
        # server_process, host, port=InferAgent.init_server(meta_model_dic, formatter)

        infer_model_params={k:v for k,v in model_params.items() if v is not None}
        query_pkl=pjoin(meta_model_dic["model_folder"], "query.pkl")
        
        infer_results=[]
        for i in tqdm(range(len(query_text_lst))):
            query_text=query_text_lst[i]
            with open(query_pkl, 'wb') as file:
                pickle.dump({"query_text": query_text, 
                             "infer_parms":infer_model_params}, file)

            response = requests.get(f'http://{host}:{port}/infer', params = {"path": query_pkl})
            infer_results.append(response.json())
            if len(infer_results)==save_step or i==len(query_text_lst)-1:
                yield infer_results
                infer_results=[]
        print("Shutting down: ", server_process)
        try:
            shutdown_server(server_process)
        except Exception as e:
            print(f"Failed to shutdown server: {e}")


    
    def single_infer(self, question, input_content, output_fd=None, verbose=False, max_new_tokens=1000, max_length=None, few_shot=True, re_try=0, model_params=None, formatter=None):
        self.initialize()

        _model_params = {} if model_params is None else model_params.copy()
        _model_params["max_length"]=max_length
        _model_params["max_new_tokens"]=max_new_tokens 
        _model_params["do_sample"]=True
        _model_params["retry"]=re_try
        infer_model_params={k:v for k,v in _model_params.items() if v is not None}


        query_template=self._plain_query_template if not few_shot or self._few_shot_query_template is None else self._few_shot_query_template
        
        _meta_model_dic=copy.deepcopy(self.meta_model_dic)
        _meta_model_dic["query_template"]=query_template
        ans=list(InferAgent.api_infer(_meta_model_dic, [{"question": question, "input":input_content}], verbose=verbose, model_params=infer_model_params, formatter=formatter))
        
        if output_fd is not None:
            pd.DataFrame(ans).to_csv(pjoin(output_fd, f"single_infer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)

        return ans[0]
    
    def list_infer(self, infer_samples, question=None, output_fd=None, verbose=False, max_new_tokens=1000, max_length=None, few_shot=True, model_params=None, parallel=False, save_step=5, resume=True, re_try=0, formatter=None):
        self.initialize()
        infer_spl_lst=self.dataset_creator.create_dataset(infer_samples, question=question)

        intermediates_folder=pjoin(self.meta_model_dic["model_folder"], "infer_intermediates")
        
        _model_params = {} if model_params is None else model_params.copy()
        _model_params["max_length"]=max_length
        _model_params["max_new_tokens"]=max_new_tokens
        _model_params["do_sample"]=True
        _model_params["retry"]=re_try
        infer_model_params={k:v for k,v in _model_params.items() if v is not None}


        res=[]
        if resume and os.path.exists(intermediates_folder):
            res=InferAgent.load_prgress_result(intermediates_folder)
            infer_spl_lst=infer_spl_lst[len(res):]
            if len(infer_spl_lst)==0:
                print("Task Already Finished.")
                return res
        else:
            os.makedirs(intermediates_folder, exist_ok=True)

        query_template=self._plain_query_template if not few_shot or self._few_shot_query_template is None else self._few_shot_query_template
        batch_query=[]
        for i in tqdm(range(len(infer_spl_lst))):
            spl=infer_spl_lst[i]
            batch_query.append(spl)
        
        _meta_model_dic=copy.deepcopy(self.meta_model_dic)
        _meta_model_dic["query_template"]=query_template
        ans_iter=InferAgent.api_infer(_meta_model_dic, batch_query, verbose=verbose, model_params=infer_model_params, formatter=formatter, save_step=5)
        for bathc_infer_res in ans_iter:
            res.extend(bathc_infer_res)                
            save_to_pkl(os.path.join(intermediates_folder, f"progress.pkl"), res)


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
        
    def infer(self, question, input_content, output_fd=None, verbose=False, max_new_tokens=1000, max_length=None, few_shot=True, re_try=0):
        self.initialize(load_model=True)
        self.load_model()

        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # bnb_8bit_use_double_quant=True,
        # bnb_8bit_quant_type="nf4",
        # bnb_8bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.meta_model_dic["model_folder"], quantization_config=bnb_config, device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.meta_model_dic["model_folder"])

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
    def load_prgress_result(folder_path):
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

        return concatenated_list
        

    def batch_infer(self, infer_samples, question=None, output_fd=None, verbose=False, max_new_tokens=1000, max_length=None, few_shot=True, parallel=False, save_step=5, resume=True, re_try=0, formatter=None):
        infer_spl_lst=self.dataset_creator.create_dataset(infer_samples, question=question)

        intermediates_folder=pjoin(self.meta_model_dic["model_folder"], "infer_intermediates")

        res=[]
        if not parallel:
            if resume and os.path.exists(intermediates_folder):
                infer_spl_lst=InferAgent.load_prgress_result(intermediates_folder)
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
