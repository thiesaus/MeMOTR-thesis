from typing import List, Sequence
from torch import Tensor
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextConfig
from transformers import CLIPTextModelWithProjection as CLIPTP
import itertools


# Borrow from YOLO-World
class HuggingCLIPLanguageBackbone(nn.Module):
    def __init__(self,
                 model_name: str = 'openai/clip-vit-base-patch32',
                 frozen_modules: Sequence[str] = ['all'],
                 dropout: float = 0.0,
                 training_use_cache: bool = False) -> None:
        super(HuggingCLIPLanguageBackbone, self).__init__()
        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        clip_config = CLIPTextConfig.from_pretrained(model_name,
                                                     attention_dropout=dropout)
        self.model = CLIPTP.from_pretrained(model_name, config=clip_config)
        self._freeze_modules()

    def forward_cache(self, text: List[List[str]]) -> Tensor:
        if not hasattr(self, "cache"):
            self.cache = self.forward_text(text)
        return self.cache

    def forward(self, text: List[List[str]]) -> Tensor:
        if self.training:
            return self.forward_text(text)
        else:
            return self.forward_cache(text)

    def forward_tokenizer(self, texts):
        if not hasattr(self, 'text'):
            text = list(itertools.chain(*texts))
            # print(text)
            # # text = ['a photo of {}'.format(x) for x in text]
            text = self.tokenizer(text=text, return_tensors='pt', padding=True)
            # print(text)
            self.text = text.to(device=self.model.device)
        return self.text

    def forward_text(self, text: List[List[str]]) -> Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences not equal in batch')
        # print(max([[len(t.split(' ')) for t in tt] for tt in text]))
        # print(num_per_batch, max(num_per_batch))
        text = list(itertools.chain(*text))
        # print(text)
        # text = ['a photo of {}'.format(x) for x in text]
        # text = self.forward_tokenizer(text)
        text = self.tokenizer(text=text, return_tensors='pt', padding=True)
        text = text.to(device=self.model.device)
        txt_outputs = self.model(**text)
        # txt_feats = txt_outputs.last_hidden_state[:, 0, :]
        txt_feats = txt_outputs.text_embeds
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        txt_feats = txt_feats.reshape(-1, num_per_batch[0],
                                      txt_feats.shape[-1])
        return txt_feats

    def _freeze_modules(self):

        if len(self.frozen_modules) == 0:
            # not freeze
            return
        if self.frozen_modules[0] == "all":
            self.model.eval()
            for _, module in self.model.named_modules():
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            return
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()

    @property
    def prompt_dim(self):
        return self.model.text_projection.out_features


DEFAULT_CLASSES = """\
person, bicycle, car, motorcycle, airplane, bus, train, truck, \
boat, traffic light, fire hydrant, stop sign, parking meter, bench, \
bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, \
backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, \
sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, \
tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, \
apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, \
chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, \
remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, \
book, clock, vase, scissors, teddy bear, hair drier, toothbrush"""

def build(model_name: str = 'openai/clip-vit-base-patch32',
          frozen_modules: Sequence[str] = ['all'],
          dropout: float = 0.0,
          training_use_cache: bool = False) -> HuggingCLIPLanguageBackbone:
    model = HuggingCLIPLanguageBackbone(model_name, frozen_modules, dropout, training_use_cache)
    # model.train()
    return model