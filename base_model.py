import pandas as pd
import random
import re
import json
import datetime
import os
import ast
import shutil
from os.path import join as pjoin
from .task_management import TaskManager
from .infer_agent import InferAgent

class BaseModel(TaskManager, InferAgent):
    def create_model(self):
        tkn={
                "model_name": self.model_name,
                "model_type": self.model_type,
                "base_model_name": self.base_model_name,
                "model_folder": self.model_folder_dic[self.base_model_name],
                "query_file": pjoin(self.task_folder, "data", "few_shot_query.txt"),
                "plain_query_file": pjoin(self.task_folder, "data", "plain_query.txt")
            }
        with open(pjoin(self.task_folder, "mdl.tkn"), "w") as f:
            json.dump(tkn, f, indent=4)
        
        self.meta_model_dic=tkn
    
    def load_model(self):
        with open(pjoin(self.task_folder, "mdl.tkn"), "r") as f:
            self.meta_model_dic=json.load(f)