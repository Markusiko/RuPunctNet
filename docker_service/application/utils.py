import transformers


def get_corrected_sentence(punct_corrector: transformers.pipeline, text: str) -> str: 
    """
    Выдает прогноз punct_corrector для строки text
    """
    
    new_text = ''
    tokens_predicted = punct_corrector(text, aggregation_strategy='simple')
    current_idx = 0
    for i in range(len(tokens_predicted)):
        if (i != len(tokens_predicted) - 1):
            if(tokens_predicted[i+1]['word'].startswith('##')):
                continue
        new_text += text[current_idx:tokens_predicted[i]['end']] + tokens_predicted[i]['entity_group']
        current_idx = tokens_predicted[i]['end']
        
    return new_text
