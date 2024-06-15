# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms
from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast

from utils import box_ops, get_tokenizer
from utils.nested_tensor import NestedTensor
from utils.nested_tensor import tensor_list_to_nested_tensor as nested_tensor_from_tensor_list
from utils.utils import (
    is_distributed, 
    distributed_world_size, 
    inverse_sigmoid, 
    accuracy, 
    interpolate
)

from utils.gdino_utils import get_phrases_from_posmap
from utils.gdino_visualizer import COCOVisualizer
from utils.gdino_vl_utils import create_positive_map_from_span

# from ..registry import MODULE_BUILD_FUNCS
# from .backbone import build as build_backbone
from .my_backbone.backbone import build as build_backbone
from .query_updater import build_gdino_style as build_query_updater
from .gdino_bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .gdino_transformer import build_transformer
from .gdino_utils import MLP, ContrastiveEmbed, sigmoid_focal_loss


# (new)
from structures.track_instances import TrackInstances


class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        query_updater, # TODO
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
        use_dab: bool = False,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = 256
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # (new) tracking
        self.use_dab = use_dab
        if self.use_dab:
            self.det_anchor = nn.Parameter(torch.randn(self.num_queries, 4))  # (N_det, 4)
            self.det_query_embed = nn.Parameter(torch.randn(self.num_queries, self.hidden_dim))       # (N_det, C)
        else:
            self.det_query_embed = nn.Parameter(torch.randn(self.num_queries, self.hidden_dim))   # (N_det, 2C)
        
        self.query_updater = query_updater
        self.det_ref_proj = nn.Linear(self.hidden_dim, 4)  # (N_det, 4)

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.tokenizer = get_tokenizer.get_tokenizer(text_encoder_type)
        self.bert = get_tokenizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)

        # (new) add Image-Text Matching (ITM) layer (like BLIP)
        self.itm_head = nn.Linear(self.bert.config.hidden_size, 2)

        # (new) add visualizer
        self.visualize = True

        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        # freeze

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # init det_ref_proj
        nn.init.xavier_uniform_(self.det_ref_proj.weight, gain=1)
        nn.init.constant_(self.det_ref_proj.bias, 0)


    def set_image_tensor(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        self.features, self.poss = self.backbone(samples)

    def unset_image_tensor(self):
        if hasattr(self, 'features'):
            del self.features
        if hasattr(self,'poss'):
            del self.poss 

    def set_image_features(self, features , poss):
        self.features = features
        self.poss = poss

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    # (new) for tracking
    def get_det_ref_pts(self):
        """ return (batch_size, Nd, 4) tensor """
        return self.det_ref_proj(self.det_query_embed)

    def get_track_ref_pts(self, tracks: list[TrackInstances]):
        """ return (batch_size, Nq, 4) tensor """
        max_len = max([len(t.ref_pts) for t in tracks])
        ref = torch.zeros((len(tracks), max_len, 4))
        for i in range(len(tracks)):
            ref[i, :len(tracks[i].ref_pts)] = tracks[i].ref_pts
        return ref

    def get_reference_points(self, tracks: list[TrackInstances], init_det_ref):
        """ return (batch_size, Nd + Nq, 4) tensor """
        det_ref = self.get_det_ref_pts().repeat(len(tracks),1,1)
        # det_ref = init_det_ref.repeat(len(tracks),1,1)
        if det_ref.shape[-1] == 2:
            det_ref = torch.cat(
                (det_ref, torch.zeros_like(det_ref, device=det_ref.device)),
                dim=-1
            )
        track_ref = self.get_track_ref_pts(tracks=tracks).to(det_ref.device)
        return torch.cat((det_ref, track_ref), dim=1)

    def get_track_query_emb(self, tracks: list[TrackInstances]):
        """ return (batch_size, Nq, hidden_dim) tensor """
        max_len = max([len(t.query_embed) for t in tracks])
        ref = torch.zeros((len(tracks), max_len, self.hidden_dim)) # TODO: careful to think about *2 or not
        for i in range(len(tracks)):
            ref[i, :len(tracks[i].query_embed), :] = tracks[i].query_embed
        return ref

    def get_query_embed(self, tracks: list[TrackInstances]):
        """ return (batch_size, Nd + Nq, hidden_dim) tensor"""
        det_query_emb = self.det_query_embed.repeat(len(tracks),1,1)
        track_query_emb = self.get_track_query_emb(tracks=tracks).to(det_query_emb.device)
        return torch.cat((det_query_emb, track_query_emb), dim=1)
        
    def get_query_mask(self, tracks: list[TrackInstances]):
        """ return (batch_size, Nd + Nq) tensor"""
        track_max_len = max([len(t.query_embed) for t in tracks])
        det_query_mask = torch.zeros((len(tracks), self.num_queries)).to(torch.bool)
        track_query_mask = torch.zeros((len(tracks), track_max_len))
        for i in range(len(tracks)):
            if len(tracks[i].query_embed) > 0:
                track_query_mask[i, len(tracks[i].query_embed):] = 1
        track_query_mask = track_query_mask.to(torch.bool)
        return torch.cat((det_query_mask, track_query_mask),dim=1).to(self.det_query_embed.device)
    
    # End of new for tracking

    def forward(self, samples: NestedTensor, targets: List = None, tracks: list[TrackInstances] = None, **kw):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.masks: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        device = samples.tensors.device

        # TODO: what is `targets`
        if targets is None:
            captions = kw["captions"]
        else:
            captions = [t["caption"] for t in targets]

        # (new) get `local_images`
        self.local_images = kw.get('local_images', None) # cuz samples in (bs, c, w, h) -> local_images in (bs, num_boxes, 3, 224, 224)

        ###### Start of Text Encoding
        tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(device)
        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        ##### Extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            # import ipdb; ipdb.set_trace()
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }

        #### End of text encoding

        #### Start of image encoding
        # import ipdb; ipdb.set_trace()
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        if not hasattr(self, 'features') or not hasattr(self, 'poss'):
            self.set_image_tensor(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(self.features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](self.features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.masks
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                self.poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None

        # (new)
        # tracks=[] # TODO
        reference_points = self.get_reference_points(tracks=tracks, init_det_ref=input_query_bbox).to(device)
        query_embed = self.get_query_embed(tracks=tracks).to(device) # i.e. target
        query_mask = self.get_query_mask(tracks=tracks).to(device)  


        ## Feed to Transformer
        hs, reference, hs_enc, ref_enc, init_box_proposal, inter_queries = self.transformer(
            srcs=srcs, 
            masks=masks, 
            pos_embeds=self.poss, 
            refpoint_embed=reference_points, # (old) input_query_bbox
            tgt=query_embed, # (old) input_query_label, 
            attn_mask=attn_mask,
            text_dict=text_dict,
            query_mask=query_mask, # (new)
        )
        # hs (list[Tensor]): (n_dec_layers=6, bs, 2*N, C)
        # reference (list[Tensor]): (n_dec_layers+1=7, bs, 2*N, 4)
        # hs_enc: (n_enc_layers=1, bs, N, C)
        # ref_enc: (n_enc_layers=1, bs, N, 4)
        # init_box_proposal: (bs, N, 4)


        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        ) # (n_dec_layers, bs, num_queries, C)

        if self.visualize:
            import os
            os.makedirs("./outputs/visualize_tmp/gdino/", exist_ok=True)
            for lid, layer in enumerate(outputs_class):
                for bs in range(layer.shape[0]):
                    torch.save(layer[bs].cpu(), f"./outputs/visualize_tmp/gdino/cls_{lid}_{bs}.tensor")

        # (new) ITM head
        itm_logits = self.itm_head(bert_output["last_hidden_state"][:, 0, :])  # (bs, 2)

        out = {
            "outputs": hs[-1],                                      # (new) (bs, Nd+Nq, C)
            "pred_logits": outputs_class[-1], 
            "pred_boxes": outputs_coord_list[-1],
            "query_mask": query_mask,                               # (new) (bs, Nd+Nq)
            "det_query_embed": query_embed[0][:self.num_queries],   # (new) (Nd, C) 
            "last_ref_pts": inverse_sigmoid(reference[-2]),         # (new) (bs, Nd+Nq, 4)
            "init_ref_pts": inverse_sigmoid(reference[0]),          # (new) (bs, Nq+Nq, 4)
            "itm_logits": itm_logits,                               # (new) (bs, 2) (ITM head)
        }

        # (new) ref: https://github.com/longzw1997/Open-GroundingDino/blob/main/models/GroundingDINO/groundingdino.py
        # Used to calculate losses
        bs, len_td = text_dict['text_token_mask'].shape
        out['text_mask'] = torch.zeros(bs, self.max_text_len, dtype=torch.bool).cuda()
        for b in range(bs):
            for j in range(len_td):
                if text_dict['text_token_mask'][b][j] == True:
                    out['text_mask'][b][j] = True

        out['token'] = tokenized

        # for debugging
        out['tokenizer'] = self.tokenizer

        # for intermediate outputs
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list, queries=inter_queries, query_mask=query_mask)

        # (new) add `text_mask` and `token` in out['aux_outputs]
        for aux in out['aux_outputs']:
            aux['text_mask'] = out['text_mask']
            aux['token'] = out['token']

        # # for encoder output
        # if hs_enc is not None:
        #     # prepare intermediate outputs
        #     interm_coord = ref_enc[-1]
        #     interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
        #     out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
        #     out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}
        unset_image_tensor = kw.get('unset_image_tensor', True)
        if unset_image_tensor:
            self.unset_image_tensor() ## If necessary
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, queries, query_mask):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b, "queries": q, "query_mask": query_mask}
            for a, b, q in zip(outputs_class[:-1], outputs_coord[:-1], queries[1:])
        ]
    
    def postprocess_single_frame(
        self, 
        previous_tracks: list[TrackInstances],
        new_tracks: list[TrackInstances],
        unmatched_dets: list[TrackInstances],
        no_augment: bool = False,
    ):
        # TODO
        return self.query_updater(previous_tracks, new_tracks, unmatched_dets, no_augment)


# @MODULE_BUILD_FUNCS.registe_with_name(module_name="groundingdino")
def build_groundingdino(args):

    backbone = build_backbone(args)
    transformer = build_transformer(args)
    query_updater = build_query_updater(args) # TODO: check again

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        query_updater=query_updater, # TODO
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
    )

    return model

