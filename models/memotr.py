# @Author       : Ruopeng Gao
# @Date         : 2022/9/4
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from .mlp import MLP
from .ffn import FFN
from .backbone import BackboneWithPE
from .deformable_transformer import DeformableTransformer
from .query_updater import build as build_query_updater
from .utils import get_clones, pos_to_pos_embed

from .backbone import build as build_backbone_with_pe
from .deformable_transformer import build as build_deformable_transformer

from utils.nested_tensor import NestedTensor
from structures.track_instances import TrackInstances
from utils.utils import inverse_sigmoid

from torch.utils.checkpoint import checkpoint

# text encoder
from .text_encoder import build as build_text_encoder
import clip

# vi-lang components
from .vi_lang_modules import VisionLanguageFusionModule, PositionEmbeddingSine1D, FeatureResizer
from einops import rearrange
import copy



class MeMOTR(nn.Module):
    def __init__(self, backbone: BackboneWithPE, transformer: DeformableTransformer,
                 query_updater: nn.Module, model_name, text_encoder,
                 num_classes: int, n_det_queries: int, n_feature_levels: int,
                 hidden_dim: int, ffn_dim: int, dropout: float,
                 aux_loss: bool = True, with_box_refine: bool = True,
                 use_checkpoint: bool = False, checkpoint_level: int = 2,
                 use_dab: bool = False,
                 visualize: bool = False,
                 **kwargs):
        super(MeMOTR, self).__init__()

        self.num_classes = num_classes
        self.n_det_queries = n_det_queries
        self.n_feature_levels = n_feature_levels
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.use_checkpoint = use_checkpoint
        self.checkpoint_level = checkpoint_level
        self.use_dab = use_dab
        self.visualize = visualize

        # # referring branch
        # self.refer_embed = nn.Linear(hidden_dim, 1)

        self.model_name = model_name
        if model_name == 'roberta-base':
            self.tokenizer, self.text_encoder = text_encoder
            self.txt_proj = FeatureResizer(
                input_feat_size=self.text_encoder.config.hidden_size,
                output_feat_size=hidden_dim,
                dropout=True,
            )
            self.fusion_module = VisionLanguageFusionModule(d_model=hidden_dim, nhead=8)
            self.text_pos = PositionEmbeddingSine1D(hidden_dim, normalize=True)

        elif model_name == 'RN50':
            self.clip = text_encoder


        # Net:
        self.backbone = backbone
        self.transformer = transformer
        self.query_updater = query_updater

        self.class_embed = nn.Linear(in_features=self.hidden_dim, out_features=num_classes)
        self.bbox_embed = MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=4, num_layers=3)
        if self.use_dab:
            self.det_anchor = nn.Parameter(torch.randn(self.n_det_queries, 4))  # (N_det, 4)
            self.det_query_embed = nn.Parameter(torch.randn(self.n_det_queries, self.hidden_dim))       # (N_det, C)
        else:
            self.det_query_embed = nn.Parameter(torch.randn(self.n_det_queries, self.hidden_dim * 2))   # (N_det, 2C)
        assert self.n_feature_levels > 1
        n_backbone_inter_layers = backbone.n_inter_layers()
        n_backbone_inter_channels = backbone.n_inter_channels()
        feature_proj_list = []
        for i in range(n_backbone_inter_layers):
            feature_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels=n_backbone_inter_channels[i], out_channels=self.hidden_dim, kernel_size=1),
                nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim)
            ))
        for _ in range(self.n_feature_levels - n_backbone_inter_layers):
            feature_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels=n_backbone_inter_channels[-1], out_channels=self.hidden_dim,
                          kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim)
            ))
        self.feature_projs = nn.ModuleList(feature_proj_list)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        # self.refer_embed.bias.data = torch.ones(1) * bias_value
        for proj in self.feature_projs:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers # because it is not two-stage
        if self.with_box_refine:
            self.class_embed = get_clones(self.class_embed, self.transformer.get_n_dec_layers())
            self.bbox_embed = get_clones(self.bbox_embed, self.transformer.get_n_dec_layers())
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.set_refine_bbox_embed(self.bbox_embed)
            # self.refer_embed = nn.ModuleList([copy.deepcopy(self.refer_embed) for i in range(num_pred)])
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(self.transformer.get_n_dec_layers())])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(self.transformer.get_n_dec_layers())])
            # self.refer_embed = nn.ModuleList([self.refer_embed for _ in range(num_pred)])

    def forward_text(self,
                     text_queries: List[str],
                     device: torch.device):
        tokenized_queries = self.tokenizer.batch_encode_plus(text_queries, padding="longest", return_tensors='pt').to(device)
        # with torch.inference_mode(mode=self.freeze_text_encoder):
        encoded_text = self.text_encoder(**tokenized_queries)
        # Transpose memory because pytorch's attention expects sequence first
        text_features = encoded_text.last_hidden_state.clone()
        text_features = self.txt_proj(text_features)  # change text embeddings dim to model dim
        # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
        text_pad_mask = tokenized_queries.attention_mask.ne(1).bool()  # [B, S]
        text_sentence_features = text_features
        return text_features, text_pad_mask, text_sentence_features

    def forward(self, 
                frame: NestedTensor, 
                tracks: list[TrackInstances],
                sentences = None):
        if self.visualize:
            os.makedirs("./outputs/visualize_tmp/memotr/", exist_ok=True)

        # backbone
        if self.use_checkpoint and self.checkpoint_level != 3:
            features, pos = checkpoint(self.backbone, frame, use_reentrant=False)
        else:
            features, pos = self.backbone(frame)

        # extract linguistic features
        if self.model_name == 'roberta-base':
            src, mask = features[-1].decompose()
            assert mask is not None
            text_word_features, text_word_mask, text_sentence_features = self.forward_text(sentences, src.device)
            # text_word_features = text_word_features.flatten(0, 1).unsqueeze(0)
            # text_word_mask = text_word_mask.flatten(0, 1).unsqueeze(0)
            text_pos = self.text_pos(NestedTensor(text_word_features, text_word_mask)).permute(2, 0, 1)
            text_word_features = text_word_features.permute(1, 0, 2)
        elif self.model_name == 'RN50':
            text_word_features = self.clip.encode_text(clip.tokenize(sentences).to(features[-1].tensors.device))
            text_word_features = text_word_features.unsqueeze(0)
            # change from FP16 to FP32
            text_word_features = text_word_features.float()

        srcs, masks = [], []

        # vi-lang early fusion (in RMOT)
        # for layer, feat in enumerate(features):
        #     src, mask = feat.decompose()

        #     # add textual processing steps
        #     src_proj_l = self.feature_projs[layer](src)
        #     n, c, h, w = src_proj_l.shape
        #     # vision-language fusion
        #     src_proj_l = rearrange(src_proj_l, 'b c h w -> (h w) b c')
        #     src_proj_l = self.fusion_module(tgt=src_proj_l,
        #                                     memory=text_word_features,
        #                                     memory_key_padding_mask=text_word_mask,
        #                                     pos=text_pos,
        #                                     query_pos=None)
        #     src_proj_l = rearrange(src_proj_l, '(h w) b c -> b c h w', h=h, w=w)

        #     srcs.append(src_proj_l)
        #     masks.append(mask)

        # vanilla 
        for layer, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.feature_projs[layer](src))
            masks.append(mask)

        if self.n_feature_levels > len(srcs):
            srcs_len = len(srcs)

            ## vi-lang early fusion (in RMOT)
            # for layer in range(srcs_len, self.n_feature_levels):
            #     if layer == srcs_len:
            #         src = self.feature_projs[layer](features[-1].tensors)
            #     else:
            #         src = self.feature_projs[layer](srcs[-1])
            #     mask = frame.masks
            #     mask = F.interpolate(mask[None, ...].float(), size=src.shape[-2:])[0].to(torch.bool)
            #     pos.append(self.backbone.position_embedding(NestedTensor(src, mask)).to(src.device))

            #     # add similar textual processing steps
            #     n,c,h,w = src.shape
            #     src = rearrange(src, 'b c h w -> (h w) b c')
            #     src = self.fusion_module(tgt=src,
            #                              memory=text_word_features,
            #                              memory_key_padding_mask=text_word_mask,
            #                              pos=text_pos,
            #                              query_pos=None)
            #     src = rearrange(src, '(h w) b c -> b c h w', h=h, w=w)
            #     srcs.append(src)
            #     masks.append(mask)

            # vanilla
            for layer in range(srcs_len, self.n_feature_levels):
                if layer == srcs_len:
                    src = self.feature_projs[layer](features[-1].tensors)
                else:
                    src = self.feature_projs[layer](srcs[-1])
                mask = frame.masks
                mask = F.interpolate(mask[None, ...].float(), size=src.shape[-2:])[0].to(torch.bool)
                pos.append(self.backbone.position_embedding(NestedTensor(src, mask)).to(src.device))
                srcs.append(src)
                masks.append(mask)
        # srcs is n_feature_levels * [(B, C, H, W)]
        # masks is n_feature_levels * [(B, H, W)]
        # pos is n_features_levels * [(B, C, H, W)]

        reference_points = self.get_reference_points(tracks=tracks).to(srcs[0].device)      # (B, Nd+Nq, 2/4)
        query_embed = self.get_query_embed(tracks=tracks).to(srcs[0].device)
        query_mask = self.get_query_mask(tracks=tracks).to(srcs[0].device)                  # (B, Nd+Nq)

        # DETR:
        outputs, init_reference, inter_references, inter_queries = self.transformer(
            srcs=srcs,
            masks=masks,
            pos_embeds=pos,
            query_embed=query_embed,
            prompt_embed=None, # text_sentence_features,
            ref_pts=reference_points,
            query_mask=query_mask
        )
        # outputs: (n_dec_layers, B, Nd+Nq, C)
        # init_reference: (B, Nd+Nq, 2)
        # inter_references: (n_dec_layers, B, Nd+Nq, 4)
        output_classes, output_bboxes = [], []
        assert outputs.ndim == 4, f"Deformable Transformer's outputs should have shape (n_dec_layers, B, Nd+Nq, C, " \
                                  f"but get n_dim={outputs.ndim}"
        for level in range(outputs.shape[0]):
            if level == 0:
                reference = init_reference
            else:
                reference = inter_references[level - 1]
            reference = inverse_sigmoid(reference)
            output_class = self.class_embed[level](outputs[level])
            bbox_tmp = self.bbox_embed[level](outputs[level])
            # output_refer = self.refer_embed[level](outputs[level])
            if reference.shape[-1] == 4:
                bbox_tmp += reference
            else:
                assert reference.shape[-1] == 2, f"Reference should have only 2 coord, but get {reference.shape[-1]}."
                bbox_tmp[..., :2] += reference
            output_bbox = bbox_tmp.sigmoid()
            output_classes.append(output_class)
            output_bboxes.append(output_bbox)

            if self.visualize:
                torch.save(reference[0, :self.n_det_queries, :].cpu(),
                           f"./outputs/visualize_tmp/memotr/detection_ref_pts_layer_{level}.tensor")
                torch.save(reference[0, self.n_det_queries:, :].cpu(),
                           f"./outputs/visualize_tmp/memotr/track_ref_pts_layer_{level}.tensor")
                torch.save(output_class[0, :self.n_det_queries, :].cpu(),
                           f"./outputs/visualize_tmp/memotr/detection_logits_layer_{level}.tensor")
                torch.save(output_class[0, self.n_det_queries:, :].cpu(),
                           f"./outputs/visualize_tmp/memotr/track_logits_layer_{level}.tensor")
                torch.save(output_bbox[0, :self.n_det_queries, :].cpu(),
                           f"./outputs/visualize_tmp/memotr/detection_boxes_layer_{level}.tensor")
                torch.save(output_bbox[0, self.n_det_queries:, :].cpu(),
                           f"./outputs/visualize_tmp/memotr/track_boxes_layer_{level}.tensor")

        output_classes = torch.stack(output_classes, dim=0)
        output_bboxes = torch.stack(output_bboxes, dim=0)
        
        res = {
            "pred_logits": output_classes[-1],
            "pred_boxes": output_bboxes[-1],
            "last_ref_pts": inverse_sigmoid(inter_references[-2, :, :, :]) if self.use_dab       # (B, Nd+Nq, 4)
            else inverse_sigmoid(inter_references[-2, :, :, :]),                                 # (B, Nd+Nq, 2)
            "query_mask": query_mask,                   # (B, Nd+Nq)
            "det_query_embed": query_embed[0][:self.n_det_queries],
            "init_ref_pts": inverse_sigmoid(init_reference)
        }
        if self.aux_loss:
            res["aux_outputs"] = self.set_aux_loss(output_classes=output_classes,
                                                   output_bboxes=output_bboxes,
                                                   query_mask=query_mask,
                                                   queries=inter_queries)
        res["outputs"] = outputs[-1]     # (B, Nd+Nq, C)
        return res

    @torch.jit.unused
    def set_aux_loss(self, output_classes, output_bboxes, query_mask, queries):
        """
        this is a workaround to make torchscript happy, as torchscript
        doesn't support dictionary with non-homogeneous values, such
        as a dict having both a Tensor and a list.
        """
        return [
            {"pred_logits": a, "pred_boxes": b, "query_mask": query_mask, "queries": c}
            for a, b, c in zip(output_classes[:-1], output_bboxes[:-1], queries[1:])
        ]

    def get_det_reference_points(self) -> torch.Tensor:
        """
        Returns: (Nd, 2)
        """
        if self.use_dab:
            return self.det_anchor
        else:
            return self.transformer.reference_points(self.det_query_embed[:, :self.hidden_dim])

    def get_track_reference_points(self, tracks: list[TrackInstances]):
        """
        Returns: (B, Nq, 2/4)
        """
        max_len = max([len(t.ref_pts) for t in tracks])
        if self.use_dab:
            references = torch.zeros((len(tracks), max_len, 4))
        else:
            # references = torch.zeros((len(tracks), max_len, 2))
            references = torch.zeros((len(tracks), max_len, 4))
        for i in range(len(tracks)):
            references[i, :len(tracks[i].ref_pts), :] = tracks[i].ref_pts
        return references

    def get_track_query_embed(self, tracks: list[TrackInstances]):
        """
        Returns: (B, Nq, 2C)
        """
        max_len = max([len(t.query_embed) for t in tracks])
        if self.use_dab:
            query_embed = torch.zeros((len(tracks), max_len, self.hidden_dim))
        else:
            query_embed = torch.zeros((len(tracks), max_len, self.hidden_dim * 2))
        for i in range(len(tracks)):
            query_embed[i, :len(tracks[i].query_embed), :] = tracks[i].query_embed
        return query_embed

    def get_reference_points(self, tracks: list[TrackInstances]):
        det_references = self.get_det_reference_points().repeat(len(tracks), 1, 1)                      # (B, Nd, 2)
        if det_references.shape[-1] == 2:
            det_references = torch.cat(
                (det_references, torch.zeros_like(det_references, device=det_references.device)),
                dim=-1
            )
        track_references = self.get_track_reference_points(tracks=tracks).to(det_references.device)     # (B, Nq, 2)
        return torch.cat((det_references, track_references), dim=1)

    def get_query_embed(self, tracks: list[TrackInstances]):
        """
        Returns: (B, Nd+Nq, 2C)
        """
        if self.use_dab:
            det_query_embed = self.det_query_embed
            det_query_embed = det_query_embed.repeat(len(tracks), 1, 1)
        else:
            det_query_embed = self.det_query_embed.repeat(len(tracks), 1, 1)                    # (B, Nd, 2C)
        track_query_embed = self.get_track_query_embed(tracks).to(det_query_embed.device)       # (B, Nq, 2C)
        return torch.cat((det_query_embed, track_query_embed), dim=1)

    def get_query_mask(self, tracks: list[TrackInstances]):
        """
        Returns: (B, Nd+Nq)
        """
        track_max_len = max([len(t.query_embed) for t in tracks])
        det_query_mask = torch.zeros((len(tracks), self.n_det_queries)).to(torch.bool)
        track_query_mask = torch.zeros((len(tracks), track_max_len))
        for i in range(len(tracks)):
            if len(tracks[i].query_embed) > 0:
                track_query_mask[i, len(tracks[i].query_embed):] = 1
        track_query_mask = track_query_mask.to(torch.bool)
        return torch.cat((det_query_mask, track_query_mask), dim=1).to(self.det_query_embed.device)

    def postprocess_single_frame(self, previous_tracks: List[TrackInstances],
                                 new_tracks: List[TrackInstances],
                                 unmatched_dets: List[TrackInstances] | None,
                                 no_augment: bool = False):
        """
        Query updating.
        """
        return self.query_updater(previous_tracks, new_tracks, unmatched_dets, no_augment)


def build(config: dict):
    dataset_num_classes = {
        "DanceTrack": 1,
        "SportsMOT": 1,
        "MOT17": 1,
        "MOT17_SPLIT": 1,
        "BDD100K": 8,
        "MOT17_COCO": 1,
        "MOT17PromptCOCO": 1,
    }
    assert config["DATASET"] in dataset_num_classes, f"Do not know the class num of {config['DATASET']} dataset."
    num_classes = dataset_num_classes[config["DATASET"]]

    backbone_with_pe = build_backbone_with_pe(config=config)
    deformable_transformer = build_deformable_transformer(config=config)
    query_updater = build_query_updater(config=config)
    model_name, text_encoder = build_text_encoder(config=config)
    
    return MeMOTR(
        backbone=backbone_with_pe,
        transformer=deformable_transformer,
        query_updater=query_updater,
        model_name=model_name,
        text_encoder=text_encoder,
        num_classes=num_classes,
        n_det_queries=config["NUM_DET_QUERIES"],
        n_feature_levels=config["NUM_FEATURE_LEVELS"],
        hidden_dim=config["HIDDEN_DIM"],
        ffn_dim=config["FFN_DIM"],
        dropout=config["DROPOUT"],
        aux_loss=True,
        with_box_refine=True,
        use_checkpoint=config["USE_CHECKPOINT"],
        checkpoint_level=config["CHECKPOINT_LEVEL"],
        use_dab=config["USE_DAB"],
        visualize=config["VISUALIZE"]
    )
