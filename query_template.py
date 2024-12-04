import pickle
import copy
from .few_shot import create_few_shot_query
from .dataset_creation import DatasetCreation

def _get_token_dict(model_type):
    token_dict = {
        "llama": {"bos": "<|begin_of_text|>",
                "eos": "<|eot_id|>",
                "end_of_query":"",
                "strip_char":["<|end_of_text|>", "<|eot_id|>"],
                "split_char":["[/INST]"]},
        "mistral": {"bos": "<s>",
                "eos": "</s>",
                "end_of_query":"[/INST]",
                "strip_char":["</s>"]}
    }
    return token_dict.get(model_type, {})

def create_query_template(model_type, output_path):
    token_dict=_get_token_dict(model_type)

    template_code = f"""
sample_questions = {{sample_questions}}
sample_inputs = {{sample_inputs}}
sample_answers= {{sample_answers}}

evaluated_text = '\\n'.join(
    f\"\"\"{token_dict['bos']}
[INST]
Question: {{{{sample_question}}}}
Input:
{{{{sample_input}}}}
[/INST]
{{{{sample_answer}}}}
{token_dict['eos']}\"\"\" for index, (sample_question, sample_input, sample_answer) in enumerate(zip(sample_questions, sample_inputs, sample_answers))
)
evaluated_text=evaluated_text+ f\"\"\"
{token_dict['bos']}
[INST]
Question: {{{{{{{{infer_question}}}}}}}}
Input:
{{{{{{{{infer_input}}}}}}}}
{token_dict["end_of_query"]}\"\"\"
"""

    with open(output_path, 'w') as file:
        file.write(template_code)


def create_fintune_query_template(model_type, output_path):
    token_dict=_get_token_dict(model_type)

    template_code = f"""
sample_questions = {{sample_questions}}
sample_inputs = {{sample_inputs}}
sample_answers= {{sample_answers}}

evaluated_text = '\\n'.join(
    f\"\"\"{token_dict['bos']}
[INST]
Question: {{{{sample_question}}}}
Input:
{{{{sample_input}}}}
[/INST]
{{{{sample_answer}}}}
{token_dict['eos']}\"\"\" for index, (sample_question, sample_input, sample_answer) in enumerate(zip(sample_questions, sample_inputs, sample_answers))
)
evaluated_text=evaluated_text+ f\"\"\"
{token_dict['bos']}
[INST]
Question: {{{{{{{{infer_question}}}}}}}}
Input:
{{{{{{{{infer_input}}}}}}}}
[/INST]
{{{{{{{{infer_answer}}}}}}}}
{token_dict['eos']}\"\"\"
"""

    with open(output_path, 'w') as file:
        file.write(template_code)


def clean_decode(gen_text, query, model_type=""):
    text=copy.deepcopy(gen_text).replace(query, "")
    token_dict=_get_token_dict(model_type)
    if token_dict["bos"]:
        text=text.split(token_dict["bos"])[-1]
    if token_dict["end_of_query"]:
        text=text.split(token_dict["end_of_query"])[-1]
    for char in token_dict.get("split_char",[]):
        text=text.split(char)[-1]
    text=text.replace("[INST]", "")
    for char in token_dict.get("strip_char",[]):
        text=text.replace(char, "")
    text=text.strip(" ").strip("\n").strip(" ")
    
    if not text:
        text=gen_text.replace(query, "").replace("[INST]", "")
        for char in token_dict.get("strip_char",[]):
            text=text.replace(char, "")
        text=text.strip(" ").strip("\n").strip(" ")
    return text