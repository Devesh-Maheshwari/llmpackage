import pandas as pd
import random
import re
import json
import datetime
import os
import ast
import shutil
import multiprocessing
from os.path import join as pjoin
from collections import defaultdict
from .config import config_properties
from functools import partial
from .query_template import create_query_template, create_few_shot_query, create_fintune_query_template, DatasetCreation, clean_decode
import logging
from log_system import Logger

logger = Logger()


class TaskManager:
    _instances = defaultdict(dict)
    def __new__(cls, base_model_name, model_name, **kwargs):
        if not cls._instances:
            try:
                multiprocessing.set_start_method('spawn')
            except RuntimeError:
                pass
        if not model_name in cls._instances.get(base_model_name, {}):
            instance = super(TaskManager, cls).__new__(cls)


            logger.set_entry(task_entry=f"TaskManager: {base_model_name}, {model_name}",                      
                default_folder="/home/ec2-user/logs/",
                task_type="TaskManager")


            cls._instances[base_model_name][model_name] = instance
            instance._init_task(base_model_name, model_name, **kwargs)
        else:
            print(f"Using existing instance for base_model: {base_model_name}, model_name: {model_name}")
            
        return cls._instances[base_model_name][model_name]

    def get_task_time(self):
        task_folders=sorted(os.listdir(self.root_task_folder), reverse=True)
        for folder in task_folders:
            mata_path=pjoin(self.root_task_folder, folder, "meta_info.json")
            if os.path.exists(mata_path):
                with open(mata_path, "r") as f:
                    meta_info_dic=json.load(f)
            if meta_info_dic["base_model_name"]==self.base_model_name and meta_info_dic["model_name"]==self.model_name:
                time_stamp=folder[len(self.model_type)+1:]
                return time_stamp
        return None
    
    @logger.log_system("task_management", "task_management.log", "Default", level=logging.DEBUG, time_stamp=False)
    def _init_task(self, base_model_name, model_name, examples=None, question='default_question', task_time=None, task_specifications={}, load_task=False, log=None):
        self.model_type=base_model_name.split("--")[0]
        self.base_model_name=base_model_name
        self.model_name=model_name
        self.task_time=task_time
        self.task_specifications=task_specifications
        self.root_task_folder=config_properties.root_task_folder
        self.model_folder_dic=config_properties.model_folder_dic
        self.clean_output=partial(clean_decode, model_type=self.model_type)
        self.flags={}
        self.model=None
        self.dataset_creator = DatasetCreation()
        self.log=log
        
        assert self.base_model_name in self.model_folder_dic, "Base model not found in model_folder dictionary."

        if load_task:
            if task_time is None:
                task_time=self.get_task_time()
                if task_time is None:
                    assert False, "Task not found."
                    
            task_folder=pjoin(self.root_task_folder,f"{self.model_type}_{task_time}")
            if task_time is not None and os.path.exists(task_folder):
                print("Resuming existing task folder.")
                print("Task folder: ", task_folder)
                self.task_folder=task_folder
                pass
            else:
                assert False, "Task folder not found."
        else:
            print(f"Creating new instance for base_model: {base_model_name}, model_name: {model_name}")
            self._create_task(task_time, few_shot_samples=examples, question=question)

    def _create_task(self, task_time, few_shot_samples=None, question=None):
        formatted_datetime = datetime.datetime.now().strftime('%m%d%Y_%H%M')
        task_time= task_time if task_time is not None else formatted_datetime

        task_folder=pjoin(self.root_task_folder,f"{self.model_type}_{task_time}")
        if os.path.exists(task_folder):
            user_input = input(f"{task_folder} exists! Please type 'delete' if you want to overwrite it.")
            if user_input == "delete":
                shutil.rmtree(task_folder)
            else:
                assert False, f"{task_folder} exists !!!"

        print("Task folder: ", task_folder)
        os.makedirs(task_folder)
        os.makedirs(pjoin(task_folder,"data"))

        meta_info_dic={
            "model_type": self.model_type,
            "base_model_name": self.base_model_name,
            "model_name": self.model_name,
            "task_spec": self.task_specifications,
            "created_time": formatted_datetime
        }

        with open(pjoin(task_folder, "meta_info.json"), "w") as f:
            json.dump(meta_info_dic,f, indent=4)

        tmp_path=pjoin(task_folder, "data","plain_query_template.txt")
        finetune_tmp_path=pjoin(task_folder, "data","plain_finetune_query_template.txt")
        create_query_template(self.model_type, tmp_path)
        create_fintune_query_template(self.model_type, finetune_tmp_path)

        create_few_shot_query(tmp_path, [], few_shot_query_path=pjoin(task_folder, "data", "plain_query.txt"))
        create_few_shot_query(finetune_tmp_path, [], few_shot_query_path=pjoin(task_folder, "data", "plain_finetune_query.txt"))
        
        
        if few_shot_samples is not None:
            samples=self.dataset_creator.create_dataset(few_shot_samples, question=question)
            create_few_shot_query(tmp_path, samples, few_shot_query_path=pjoin(task_folder, "data", "few_shot_query.txt"))
            create_few_shot_query(finetune_tmp_path, samples, few_shot_query_path=pjoin(task_folder, "data", "few_shot_finetune_query.txt"))
        
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        self.task_folder=task_folder

    @staticmethod
    def find_oldtasks(root_task_folder):
        task_folders = sorted(os.listdir(root_task_folder), reverse=True)
        latest_folders = {}

        # Identify the latest task folder for each model type
        for folder in task_folders:
            meta_path = pjoin(root_task_folder, folder, "meta_info.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as file:
                    meta_info = json.load(file)
                    model_name = meta_info["model_name"]
                _task_time = folder[folder.find('_')+1:]
                if latest_folders.get(model_name,("-1",None))[0]< _task_time:
                    latest_folders[model_name] = (_task_time, folder)
        keep_folder={}
        for k, v in latest_folders.items():
            keep_folder[k]=v[1]
        return list(set(task_folders)-set(keep_folder.values())), keep_folder


    @staticmethod
    def clean_history(model_type=None):
        if model_type is None:
            user_confirm = input("Please confirm to delete task folders. Type 'all' to delete all historical tasks. Type 'old' to keep the latest tasks for each model type.")
            if user_confirm.lower() == "all":
                shutil.rmtree(config_properties.root_task_folder)
                print("All task folders deleted.")
            elif user_confirm.lower() == "old":
                root_task_folder=config_properties.root_task_folder
                old_task_folders, latest_folders = TaskManager.find_oldtasks(root_task_folder)
                user_confirm = input(f"Here are the latest task to keep, {latest_folders}. Type 'delete' to continue.")
                if user_confirm.lower() == "delete":
                    for folder in old_task_folders:
                        shutil.rmtree(pjoin(root_task_folder, folder))
                    print("Old task folders deleted.")
                else:
                    print("Deletion Cancelled.")
            else:
                print("No action taken.")
        else:
            user_confirm = input(f"Please confirm to delete all task folders for {model_type}. Type 'yes' to continue...")
            if user_confirm.lower() == "all":
                task_folder_all=os.listdir(config_properties.root_task_folder)
                delete_folder_lst=[pjoin(config_properties.root_task_folder, folder) for folder in task_folder_all if folder.startswith(model_type)]
                for folder in delete_folder_lst:
                    shutil.rmtree(folder)
                print(f"All task folders for {model_type} deleted.")
            else:
                print("No action taken.")