# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
import torch

from utils.utils import distributed_rank
from .memotr import build as build_memotr
from .my_groundingdino_track import build_groundingdino
from utils.gdino_slconfig import SLConfig
from utils.gdino_utils import clean_state_dict

def build_model(config: dict):
    model = build_memotr(config=config)
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))
    return model



def build_gdino_model(config: dict): 
    args = SLConfig.fromfile(config['GDINO']['CONFIG_FILE'])
    args.device = "cuda" if not config['GDINO']['CPU_ONLY'] else "cpu"
    model = build_groundingdino(args)
    checkpoint = torch.load(config['GDINO']['CHECKPOINT_PATH'], map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(load_res)
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))
    return model
