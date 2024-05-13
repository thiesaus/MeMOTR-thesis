# text encoder
import os 

def build_roberta(config):
    """
    Returns:
        tokenizer: RobertaTokenizerFast
        text_encoder: RobertaModel
    """
    from transformers import RobertaModel, RobertaTokenizerFast
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
    
    if config['FREEZE_TEXT_ENCODER']:
        for param in text_encoder.parameters():
            param.requires_grad = False

    return tokenizer, text_encoder


def build_clip(config):
    """
    Returns:
        tokenizer: CLIPProcessor
        text_encoder: CLIPModel
    """
    import clip
    import torch
    device = config['DEVICE']
    model_name = config['MODEL_NAME']
    model, _ = clip.load(model_name, device=torch.device(device))

    # Freeze text encoder
    if config['FREEZE_TEXT_ENCODER']:
        for param in model.parameters():
            param.requires_grad = False

    return model # used via `model.encode_text(clip.tokenize(["a list of strings"]))`


def build(config):
    options = {
        'roberta-base': 'roberta-base',
        'clip-rn': 'RN50',
        'clip-vit': 'ViT-B/32',
    }
    model_name = options[config['MODEL_NAME']]
    assert model_name is not None, f"Model {config['MODEL_NAME']} not supported"

    if model_name == 'roberta-base':
        return build_roberta(config)
    
    if model_name in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']:
        return build_clip(config)
    
    raise NotImplementedError(f"Model {model_name} not implemented")