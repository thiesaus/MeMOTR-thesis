# text encoder
from transformers import RobertaModel, RobertaTokenizerFast
import os 

def build_roberta(config):
    """
    Returns:
        tokenizer: RobertaTokenizerFast
        text_encoder: RobertaModel
    """
    hf_local_path = config['HF_LOCAL_PATH']
    model_name = config['MODEL_NAME']
    if not os.path.exists(hf_local_path):
        os.makedirs(hf_local_path)
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        text_encoder = RobertaModel.from_pretrained(model_name)
        tokenizer.save_pretrained(hf_local_path)
        text_encoder.save_pretrained(hf_local_path)
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained(hf_local_path)
        text_encoder = RobertaModel.from_pretrained(hf_local_path)
    
    return tokenizer, text_encoder


def build(config):
    options = {
        'roberta-base': 'roberta-base',
    }
    model_name = options[config['MODEL_NAME']]
    assert model_name is not None, f"Model {config['MODEL_NAME']} not supported"

    if model_name == 'roberta-base':
        tokenizer, text_encoder = build_roberta(config)

    if config['FREEZE_TEXT_ENCODER']:
        for param in text_encoder.parameters():
            param.requires_grad = False

    return tokenizer, text_encoder