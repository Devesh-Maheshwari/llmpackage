from functools import singledispatch, update_wrapper
import pandas as pd
import os
import json

def methdispatch(func):
    dispatcher = singledispatch(func)
    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper

class DatasetCreation:
    @methdispatch
    def create_dataset(self, obj, output_path=None, question=None):
        raise NotImplementedError(f"Unsupported type: {type(obj)}")

    @create_dataset.register(pd.DataFrame)
    def _create_dataset_from_df(self, df, output_path=None, question=None):
        if not all(col in df.columns for col in ["input", "answer"]):
            raise ValueError("DataFrame must contain 'input' and 'answer' columns")
        
        records = []
        for _, row in df.iterrows():
            record = {
                "input": row["input"],
                "answer": row["answer"],
                "question": question if question is not None else row.get("question", None)
            }
            records.append(record)
        if output_path is not None:
            with open(output_path, "w") as f:
                json.dump(records, f, indent=4)
        else:
            return records

    @create_dataset.register(list)
    def _create_dataset_from_list(self, lst, output_path=None, question=None):
        records = []
        for item in lst:
            if not all(key in item for key in ["input", "answer"]):
                raise ValueError("Each dictionary in the list must contain 'input' and 'answer' keys")
            record = {
                "input": item["input"],
                "answer": item["answer"],
                "question": question if question is not None else item.get("question", None)
            }
            records.append(record)
        
        if output_path is not None:
            with open(output_path, "w") as f:
                json.dump(records, f, indent=4)
        else:
            return records

    @create_dataset.register(str)
    def _create_dataset_from_path(self, path, output_path=None, question=None):
        if not os.path.isdir(path):
            raise ValueError(f"Provided path is not a directory: {path}")

        input_folder = os.path.join(path, "input")
        answer_folder = os.path.join(path, "answer")

        records = []
        if os.path.exists(input_folder) and os.path.exists(answer_folder):
            input_files = {file.lower().replace("input_", ""): file for file in os.listdir(input_folder) if file.lower().startswith("input_")}
            answer_files = {file.lower().replace("answer_", ""): file for file in os.listdir(answer_folder) if file.lower().startswith("answer_")}
            
            for key, input_file in input_files.items():
                with open(os.path.join(input_folder, input_file), 'r') as infile:
                    input_content = infile.read().strip()
                answer_content = ""
                if key in answer_files:
                    with open(os.path.join(answer_folder, answer_files[key]), 'r') as ansfile:
                        answer_content = ansfile.read().strip()
                records.append({
                    "input": input_content,
                    "answer": answer_content,
                    "question": question
                })
        else:
            for filename in os.listdir(path):
                if filename.endswith('.json'):
                    with open(os.path.join(path, filename), 'r') as file:
                        data = json.load(file)
                        if not all(key in data for key in ["input", "answer"]):
                            raise ValueError(f"File {filename} must contain 'input' and 'answer' keys")

                        record = {
                            "input": data["input"],
                            "answer": data["answer"],
                            "question": question if question is not None else data.get("question", None)
                        }
                        records.append(record)
            
        if output_path is not None:
            with open(output_path, "w") as f:
                json.dump(records, f, indent=4)
        else:
            return records
