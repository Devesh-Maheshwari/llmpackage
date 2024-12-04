
def create_few_shot_query(query_templates_path, samples, few_shot_query_path=None):
    with open(query_templates_path, 'r') as file:
        template_code = "".join(file.readlines())

    filled_template = template_code.format(
        sample_questions=[str(sample["question"]).replace('{', '{{').replace('}', '}}') for sample in samples],
        sample_inputs=[str(sample["input"]).replace('{', '{{').replace('}', '}}') for sample in samples],
        sample_answers=[str(sample["answer"]).replace('{', '{{').replace('}', '}}') for sample in samples]
    )

    local_scope = {}
    exec(filled_template, globals(), local_scope)

    # Access the 'evaluated_text' variable from the local scope
    evaluated_text = local_scope.get('evaluated_text')

    if few_shot_query_path is None:
        return evaluated_text
    else:
        with open(few_shot_query_path, 'w') as file:
            file.write(evaluated_text)

