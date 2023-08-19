# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
# import random
# random.choices([1,2,3],k=2)
import random

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()


        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret


    def prepare_for_dn(self,mask_features,dn_args):
        targets,scalar,noise_scale=dn_args['tgt'],dn_args['scalar'],dn_args['noise_scale']
        boxes=[targets[i]['boxes'] for i in range(len(targets))]
        num_boxes=[len(b) for b in boxes]
        single_pad=max_num=max(num_boxes)

        if max_num==0:
            return None, self.query_feat.weight.unsqueeze(1).repeat(1, len(boxes), 1), None,None
        if scalar>=100:
            scalar=scalar//max_num
        pad_size=scalar*max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size']=pad_size
        bs=len(boxes)
        padding=torch.zeros([bs,pad_size,self.query_feat.weight.shape[-1]]).cuda()
        # padding[]
        masks=[F.interpolate(targets[i]['masks'].unsqueeze(1).float(),size=mask_features.shape[-2:],mode="bilinear") for i in range(len(targets))]
        # import pdb;pdb.set_trace()
        known_features=torch.cat([(m*mask_features[i]).flatten(-2).sum(-1)/(m.flatten(-2).sum(-1)+1e-8) for i,m in enumerate(masks)]).detach()
        known_features=known_features.repeat(scalar,1)

        dn_delta=(torch.rand_like(known_features)*2-1.0)*noise_scale*known_features
        noised_known_features=known_features+dn_delta

        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')

        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_boxes])  # [1,2, 1,2,3]
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(scalar)]).long().cuda()

        # for i in range(bs):
        padding[(known_bid,map_known_indices)]=noised_known_features

        res=self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        res=torch.cat([padding.transpose(0,1),res],dim=0)
        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return None,res ,attn_mask,dn_args_
        # pass


    def postprocess_for_dn(self,predictions_class, predictions_mask):
        n_lys=len(predictions_class)
        dn_predictions_class,predictions_class=[predictions_class[i][:,:-self.num_queries] for i in range(n_lys)], \
                                               [predictions_class[i][:,-self.num_queries:] for i in range(n_lys)]
        dn_predictions_mask, predictions_mask = [predictions_mask[i][:, :-self.num_queries] for i in range(n_lys)], \
                                                  [predictions_mask[i][:, -self.num_queries:] for i in range(n_lys)]
        return predictions_class, predictions_mask, dn_predictions_class, dn_predictions_mask
        # pass


    def forward(self, x, mask_features, mask = None,dn_args=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        if dn_args:
            query_embed,output,tgt_mask,dn_args_=self.prepare_for_dn(mask_features, dn_args)
        else:

            # query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            query_embed=None
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
            tgt_mask=None
        # import pdb;pdb.set_trace()

        predictions_class = []
        predictions_mask = []
        dn_predictions_class = []
        dn_predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        # import pdb;pdb.set_trace()
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # import pdb;pdb.set_trace()
            #Deal with extreme cases when all elements are masked
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=tgt_mask,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )
            # attn_mask:[B*h, Q, HW]
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1
        if not (tgt_mask is None):
            predictions_class, predictions_mask, dn_predictions_class, dn_predictions_mask=\
                self.postprocess_for_dn(predictions_class,predictions_mask)

            dn_out = {
                'pred_logits': dn_predictions_class[-1],
                'pred_masks': dn_predictions_mask[-1],
                'aux_outputs': self._set_aux_loss(
                    dn_predictions_class if self.mask_classification else None, dn_predictions_mask
                ),
                'dn_args':dn_args_

            }
        else:
            dn_out=None
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
            'dn_out':dn_out
        }

        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        # import pdb;pdb.set_trace()
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # import pdb;pdb.set_trace()
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoderMaskDN(nn.Module):
    _version = 2

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
            self,
            in_channels,
            mask_classification=True,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,
            mask_dim: int,
            enforce_input_project: bool,
            dn_mode="base",
            head_dn=False,
            all_lys=False,
            dn_ratio=0.5,
            dn_label_noise_ratio=-1.,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        self.head_dn=head_dn
        self.dn_ratio=dn_ratio
        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.dn_label_noise_ratio=dn_label_noise_ratio
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_classes=num_classes
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.dn_mode=dn_mode

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.matching_dict=dict()
        self.label_enc = nn.Embedding(num_classes, hidden_dim)
        self.all_lys=all_lys

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["dn_mode"] = cfg.MODEL.MASK_FORMER.DN_MODE
        ret["head_dn"] = cfg.MODEL.MASK_FORMER.HEAD_DN
        ret["all_lys"] = cfg.MODEL.MASK_FORMER.ALL_LY_DN
        ret["dn_ratio"] = cfg.MODEL.MASK_FORMER.DN_RATIO
        ret["dn_label_noise_ratio"] = cfg.MODEL.MASK_FORMER.LB_NOISE_RATIO

        return ret

    def prepare_for_normal(self,bs,mask_features,size_list):
        query_embed = None
        tgt_mask = None
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                               attn_mask_target_size=size_list[0])
        return query_embed,output,tgt_mask,outputs_class, outputs_mask, attn_mask

    def prepare_for_dn_v2(self, mask_features, dn_args,size_list):
        targets, scalar, noise_scale = dn_args['tgt'], dn_args['scalar'], dn_args['noise_scale']
        boxes = [targets[i]['boxes'] for i in range(len(targets))]
        num_boxes = [len(b) for b in boxes]
        single_pad = max_num = max(num_boxes)
        if max_num == 0 or scalar==0:
            return None
        if scalar >= 100:
            scalar = scalar // max_num
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size
        bs = len(boxes)

        padding = torch.zeros([bs, pad_size, self.query_feat.weight.shape[-1]]).cuda()
        padding_mask = torch.ones([bs, pad_size, size_list[0][0]*size_list[0][1]]).cuda().bool()
        # padding[]
        # gt_masks=torch.cat([])
        # import pdb;pdb.set_trace()
        #attn_mask[B * h, Q, HW]
        #masks [b*h,pad_size,hw]
        # masks_ = [F.interpolate(targets[i]['masks'].unsqueeze(1).float(), size=mask_features.shape[-2:], mode="bilinear")
        #          for i in range(len(targets))]
        masks = torch.cat([F.interpolate(targets[i]['masks'].unsqueeze(1), size=size_list[0], mode="nearest").flatten(1)<0.5
                 for i in range(len(targets))]).repeat(scalar, 1)

        labels=torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(scalar, 1).view(-1)
        known_labels = self.label_enc(known_labels)
        # noised_known_features = known_labels

        # known_features = torch.cat([(m * mask_features[i]).flatten(-2).mean(-1) for i, m in enumerate(masks_)])
        # known_features = known_features.repeat(scalar, 1).detach()

        dn_delta = (torch.rand_like(known_labels) * 2 - 1.0) * noise_scale * known_labels
        noised_known_features = known_labels + dn_delta

        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')

        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_boxes])  # [1,2, 1,2,3]
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(scalar)]).long().cuda()

        # for i in range(bs):
        padding[(known_bid, map_known_indices)] = noised_known_features
        padding_mask[(known_bid, map_known_indices)]=masks
        padding_mask=padding_mask.unsqueeze(1).repeat([1,8,1,1])

        res = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        res = torch.cat([padding.transpose(0, 1), res], dim=0)
        ###################

        outputs_class, outputs_mask, attn_mask_ = self.forward_prediction_heads(res, mask_features,attn_mask_target_size=size_list[0])
        attn_mask_=attn_mask_.view([bs,8,-1,attn_mask_.shape[-1]])
        attn_mask_[:,:,:-self.num_queries]=padding_mask
        attn_mask_=attn_mask_.flatten(0,1)
        #####################

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return None, res, attn_mask, dn_args_,outputs_class, outputs_mask, attn_mask_
        # pass

    def prepare_for_dn_v3(self, mask_features, dn_args,size_list):
        targets, scalar, noise_scale = dn_args['tgt'], dn_args['scalar'], dn_args['noise_scale']
        boxes = [targets[i]['boxes'] for i in range(len(targets))]
        num_boxes = [len(b) for b in boxes]
        single_pad = max_num = max(num_boxes)
        if max_num == 0 or scalar==0:
            return None
        if scalar >= 100:
            scalar = scalar // max_num
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size
        bs = len(boxes)

        padding = torch.zeros([bs, pad_size, self.query_feat.weight.shape[-1]]).cuda()
        padding_mask = torch.ones([bs, pad_size, size_list[0][0]*size_list[0][1]]).cuda().bool()

        masks = torch.cat([F.interpolate(targets[i]['masks'].float().unsqueeze(1), size=size_list[0], mode="area").flatten(1)<=1e-8
                 for i in range(len(targets)) if len(targets[i]['masks'])>0]).repeat(scalar, 1)
        head_dn=self.head_dn
        if head_dn:
            masks = masks[:, None].repeat(1, 8, 1)
            areas = (~masks).sum(2)
            noise_ratio = areas * noise_scale / (size_list[0][0] * size_list[0][1])
            delta_mask = torch.rand_like(masks, dtype=torch.float) < noise_ratio[:,:, None]
            masks = torch.logical_xor(masks, delta_mask) #[q,h,h*w]
        else:
            areas= (~masks).sum(1)
            noise_ratio=areas*noise_scale/(size_list[0][0]*size_list[0][1])
            delta_mask=torch.rand_like(masks,dtype=torch.float)<noise_ratio[:,None]
            masks=torch.logical_xor(masks,delta_mask)


        # torch.sum(dim=)

        labels=torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(scalar, 1).view(-1)
        known_labels = self.label_enc(known_labels)
        noised_known_features = known_labels

        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')

        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_boxes])  # [1,2, 1,2,3]
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(scalar)]).long().cuda()

        # for i in range(bs):
        padding[(known_bid, map_known_indices)] = noised_known_features
        if head_dn:
            padding_mask=padding_mask.unsqueeze(2).repeat([1,1,8,1]) #[bs,q,h,h*w]
            padding_mask[(known_bid, map_known_indices)] = masks
            padding_mask=padding_mask.transpose(1,2)
        else:
            padding_mask[(known_bid, map_known_indices)]=masks
            padding_mask=padding_mask.unsqueeze(1).repeat([1,8,1,1])

        res = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        res = torch.cat([padding.transpose(0, 1), res], dim=0)
        ###################

        outputs_class, outputs_mask, attn_mask_ = self.forward_prediction_heads(res, mask_features,attn_mask_target_size=size_list[0])
        attn_mask_=attn_mask_.view([bs,8,-1,attn_mask_.shape[-1]])
        attn_mask_[:,:,:-self.num_queries]=padding_mask
        attn_mask_=attn_mask_.flatten(0,1)
        #####################

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return None, res, attn_mask, dn_args_,outputs_class, outputs_mask, attn_mask_
        # pass

    def prepare_for_dn_v4(self, mask_features, dn_args,size_list):
        targets, scalar, noise_scale = dn_args['tgt'], dn_args['scalar'], dn_args['noise_scale']
        boxes = [targets[i]['boxes'] for i in range(len(targets))]
        num_boxes = [len(b) for b in boxes]
        single_pad = max_num = max(num_boxes)
        if max_num == 0 or scalar==0:
            return None
        if scalar >= 100:
            scalar = scalar // max_num
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size
        bs = len(boxes)

        padding = torch.zeros([bs, pad_size, self.query_feat.weight.shape[-1]]).cuda()
        padding_mask = torch.ones([bs, pad_size, size_list[0][0]*size_list[0][1]]).cuda().bool()
        masks = torch.cat([F.interpolate(targets[i]['masks'].float().unsqueeze(1), size=size_list[0], mode="area").flatten(1)<=1e-8
                 for i in range(len(targets)) if len(targets[i]['masks'])>0]).repeat(scalar, 1)
        if self.head_dn:
            masks = masks[:, None].repeat(1, 8, 1)
            areas = (~masks).sum(2)
            noise_ratio = areas * noise_scale / (size_list[0][0] * size_list[0][1])
            delta_mask = torch.rand_like(masks, dtype=torch.float) < noise_ratio[:,:, None]
            masks = torch.logical_xor(masks, delta_mask) #[q,h,h*w]
        else:
            areas= (~masks).sum(1)
            noise_ratio=areas*noise_scale/(size_list[0][0]*size_list[0][1])
            delta_mask=torch.rand_like(masks,dtype=torch.float)<noise_ratio[:,None]
            masks=torch.logical_xor(masks,delta_mask)


        # torch.sum(dim=)

        labels=torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(scalar, 1).view(-1)
        known_labels = self.label_enc(known_labels)
        noised_known_features = known_labels


        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')

        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_boxes])  # [1,2, 1,2,3]
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(scalar)]).long().cuda()

        # for i in range(bs):
        padding[(known_bid, map_known_indices)] = noised_known_features
        if self.head_dn:
            padding_mask=padding_mask.unsqueeze(2).repeat([1,1,8,1]) #[bs,q,h,h*w]
            padding_mask[(known_bid, map_known_indices)] = masks
            padding_mask=padding_mask.transpose(1,2)
        else:
            padding_mask[(known_bid, map_known_indices)]=masks
            padding_mask=padding_mask.unsqueeze(1).repeat([1,8,1,1])

        res = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        res = torch.cat([padding.transpose(0, 1), res], dim=0)
        ###################

        outputs_class, outputs_mask, attn_mask_ = self.forward_prediction_heads(res, mask_features,attn_mask_target_size=size_list[0])
        attn_mask_=attn_mask_.view([bs,8,-1,attn_mask_.shape[-1]])
        attn_mask_[:,:,:-self.num_queries]=padding_mask
        attn_mask_=attn_mask_.flatten(0,1)
        #####################

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return (known_bid,map_known_indices), res, attn_mask, dn_args_,outputs_class, outputs_mask, attn_mask_
        # pass

    def prepare_for_dn_v5(self, mask_features, dn_args,size_list):
        targets, scalar, noise_scale = dn_args['tgt'], dn_args['scalar'], dn_args['noise_scale']
        boxes = [targets[i]['boxes'] for i in range(len(targets))]
        num_boxes = [len(b) for b in boxes]
        single_pad = max_num = max(num_boxes)
        if scalar >= 100:
            scalar = scalar // max_num
        if max_num == 0 or scalar==0:
            return None

        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size
        bs = len(boxes)

        padding = torch.zeros([bs, pad_size, self.query_feat.weight.shape[-1]]).cuda()
        padding_mask = torch.ones([bs, pad_size, size_list[0][0]*size_list[0][1]]).cuda().bool()
        masks = torch.cat([F.interpolate(targets[i]['masks'].float().unsqueeze(1), size=size_list[0], mode="area").flatten(1)<=1e-8
                 for i in range(len(targets)) if len(targets[i]['masks'])>0]).repeat(scalar, 1)
        if self.head_dn:
            masks = masks[:, None].repeat(1, 8, 1)
            areas = (~masks).sum(2)
            noise_ratio = areas * noise_scale / (size_list[0][0] * size_list[0][1])
            delta_mask = torch.rand_like(masks, dtype=torch.float) < noise_ratio[:,:, None]
            masks = torch.logical_xor(masks, delta_mask) #[q,h,h*w]
        else:
            areas= (~masks).sum(1)
            noise_ratio=areas*noise_scale/(size_list[0][0]*size_list[0][1])
            delta_mask=torch.rand_like(masks,dtype=torch.float)<noise_ratio[:,None]
            masks=torch.logical_xor(masks,delta_mask)


        # torch.sum(dim=)

        labels=torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(scalar, 1).view(-1)
        # known_labels = self.label_enc(known_labels)

        dn_label_noise_ratio = self.dn_label_noise_ratio
        knwon_labels_expand = known_labels.clone()
        if dn_label_noise_ratio > 0:
            prob = torch.rand_like(knwon_labels_expand.float())
            chosen_indice = prob < dn_label_noise_ratio
            new_label = torch.randint_like(knwon_labels_expand[chosen_indice], 0,
                                           self.num_classes)  # randomly put a new one here
            # gt_labels_expand.scatter_(0, chosen_indice, new_label)
            knwon_labels_expand[chosen_indice] = new_label

        known_labels = self.label_enc(knwon_labels_expand)
        # noised_known_features = known_labels
        noised_known_features = known_labels


        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')
        # import pdb;pdb.set_trace()
        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_boxes])  # [1,2, 1,2,3]
        # print(map_known_indices)
        # print(boxes)
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(scalar)]).long().cuda()

        # for i in range(bs):
        padding[(known_bid, map_known_indices)] = noised_known_features
        if self.head_dn:
            padding_mask=padding_mask.unsqueeze(2).repeat([1,1,8,1]) #[bs,q,h,h*w]
            padding_mask[(known_bid, map_known_indices)] = masks
            padding_mask=padding_mask.transpose(1,2)
        else:
            padding_mask[(known_bid, map_known_indices)]=masks
            padding_mask=padding_mask.unsqueeze(1).repeat([1,8,1,1])

        res = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        res = torch.cat([padding.transpose(0, 1), res], dim=0)
        ###################

        outputs_class, outputs_mask, attn_mask_ = self.forward_prediction_heads(res, mask_features,attn_mask_target_size=size_list[0])
        attn_mask_=attn_mask_.view([bs,8,-1,attn_mask_.shape[-1]])
        attn_mask_[:,:,:-self.num_queries]=padding_mask
        attn_mask_=attn_mask_.flatten(0,1)
        #####################

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return (known_bid,map_known_indices), res, attn_mask, dn_args_,outputs_class, outputs_mask, attn_mask_
        # pass

    def prepare_for_dn_v6(self, mask_features, dn_args, size_list):
        targets, scalar, noise_scale = dn_args['tgt'], dn_args['scalar'], dn_args['noise_scale']
        boxes = [targets[i]['boxes'] for i in range(len(targets))]
        num_boxes = [len(b) for b in boxes]
        single_pad = max_num = max(num_boxes)
        if max_num == 0 or scalar == 0:
            return None
        if scalar >= 100:
            scalar = scalar // max_num
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size
        bs = len(boxes)

        padding = torch.zeros([bs, pad_size, self.query_feat.weight.shape[-1]]).cuda()
        padding_mask_3level = []
        for i in range(len(size_list)):
            padding_mask = torch.ones([bs, pad_size, size_list[i][0] * size_list[i][1]]).cuda().bool()
            padding_mask_3level.append(padding_mask)

        # padding[]
        # gt_masks=torch.cat([])
        # import pdb;pdb.set_trace()
        # attn_mask[B * h, Q, HW]
        # masks [b*h,pad_size,hw]
        # masks_ = [F.interpolate(targets[i]['masks'].unsqueeze(1).float(), size=mask_features.shape[-2:], mode="bilinear")
        #          for i in range(len(targets))]
        # True means no object
        # import ipdb; ipdb.set_trace()
        # masks0 = torch.cat([F.interpolate(targets[i]['masks'].unsqueeze(1), size=size_list[0], mode="nearest").flatten(1)<0.5
        #          for i in range(len(targets))]).repeat(scalar, 1)
        boxes = torch.cat([t['boxes'] for t in targets])  # x, y, x, y
        knwon_boxes_expand = boxes.repeat(scalar, 1)

        # knwon_labels_expand = known_labels.clone()
        dn_mask_noise_scale = noise_scale
        masks_3level = []
        if dn_mask_noise_scale > 0:
            masks = torch.cat(
                [F.interpolate(targets[i]['masks'].unsqueeze(1), size=size_list[-1], mode="nearest") < 0.5
                 for i in range(len(targets))]).flatten(0, 1)
            masks = masks.repeat(scalar, 1, 1)
            # import ipdb; ipdb.set_trace()
            diff = torch.zeros_like(knwon_boxes_expand)
            diff[..., :2] = knwon_boxes_expand[..., 2:] / 2
            diff[..., 2:] = knwon_boxes_expand[..., 2:]
            delta_masks = torch.mul((torch.rand_like(knwon_boxes_expand) * 2 - 1.0), diff) * dn_mask_noise_scale
            delta_masks *= size_list[-1][-1]
            new_masks = []
            for mask, delta_mask in zip(masks, delta_masks):
                x, y = torch.where(mask == False)
                delta_x = delta_mask[0]
                delta_y = delta_mask[1]
                # import ipdb; ipdb.set_trace()
                x = x + delta_x
                y = y + delta_y
                x = x.clamp(min=0, max=size_list[-1][-2] - 1)
                y = y.clamp(min=0, max=size_list[-1][-1] - 1)
                mask = torch.ones_like(mask)
                mask[x.long(), y.long()] = 0
                mask = mask > 0.5
                new_masks.append(mask)
            new_masks = torch.stack(new_masks)
            masks = new_masks.flatten(1)

            masks_3level.append(masks)
            # import ipdb; ipdb.set_trace()
            masks = (F.interpolate(new_masks.unsqueeze(1).float(), size=size_list[-2], mode="nearest") > 0.5).flatten(1)
            masks_3level.append(masks)

            masks = (F.interpolate(new_masks.unsqueeze(1).float(), size=size_list[0], mode="nearest") > 0.5).flatten(1)
            masks_3level.append(masks)

            # knwon_boxes_expand = knwon_boxes_expand.clamp(min=0.0, max=1.0)

        labels = torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(scalar, 1).view(-1)

        dn_label_noise_ratio = self.dn_label_noise_ratio
        knwon_labels_expand = known_labels.clone()
        if dn_label_noise_ratio > 0:
            prob = torch.rand_like(knwon_labels_expand.float())
            chosen_indice = prob < dn_label_noise_ratio
            new_label = torch.randint_like(knwon_labels_expand[chosen_indice], 0,
                                           self.num_classes)  # randomly put a new one here
            # gt_labels_expand.scatter_(0, chosen_indice, new_label)
            knwon_labels_expand[chosen_indice] = new_label

        known_labels = self.label_enc(knwon_labels_expand)
        # noised_known_features = known_labels

        # known_features = torch.cat([(m * mask_features[i]).flatten(-2).mean(-1) for i, m in enumerate(masks_)])
        # known_features = known_features.repeat(scalar, 1).detach()
        if dn_label_noise_ratio > 0:
            noised_known_features = known_labels
        else:
            dn_delta = (torch.rand_like(known_labels) * 2 - 1.0) * noise_scale * known_labels
            noised_known_features = known_labels + dn_delta

        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')

        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_boxes])  # [1,2, 1,2,3]
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(scalar)]).long().cuda()

        # for i in range(bs):
        padding[(known_bid, map_known_indices)] = noised_known_features

        for i, padding_mask in enumerate(padding_mask_3level):
            padding_mask_3level[i][(known_bid, map_known_indices)] = masks_3level[2-i]
            padding_mask_3level[i] = padding_mask_3level[i].unsqueeze(1).repeat([1, 8, 1, 1])

        res = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        res = torch.cat([padding.transpose(0, 1), res], dim=0)
        ###################

        outputs_class, outputs_mask, attn_mask_ = self.forward_prediction_heads(res, mask_features,
                                                                                attn_mask_target_size=size_list[0])
        attn_mask_ = attn_mask_.reshape([bs, 8, -1, attn_mask_.shape[-1]])
        attn_mask_[:, :, :-self.num_queries] = padding_mask_3level[0]
        attn_mask_ = attn_mask_.flatten(0, 1)
        #####################

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return None, res, attn_mask, dn_args_, outputs_class, outputs_mask, attn_mask_, padding_mask_3level
        # pass

    def prepare_for_dn_v7(self, mask_features, dn_args, size_list, shift_dn=False):
        targets, scalar, noise_scale = dn_args['tgt'], dn_args['scalar'], dn_args['noise_scale']
        boxes = [targets[i]['boxes'] for i in range(len(targets))]
        num_boxes = [len(b) for b in boxes]
        single_pad = max_num = max(num_boxes)
        if max_num == 0 or scalar == 0:
            return None
        if scalar >= 100:
            scalar = scalar // max_num
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size
        bs = len(boxes)

        padding = torch.zeros([bs, pad_size, self.query_feat.weight.shape[-1]]).cuda()
        padding_mask_3level = []
        for i in range(len(size_list)):
            padding_mask = torch.ones([bs, pad_size, size_list[i][0] * size_list[i][1]]).cuda().bool()
            padding_mask_3level.append(padding_mask)

        boxes = torch.cat([t['boxes'] for t in targets])  # x, y, x, y
        knwon_boxes_expand = boxes.repeat(scalar, 1)

        # knwon_labels_expand = known_labels.clone()
        dn_mask_noise_scale_shift = noise_scale
        dn_mask_noise_scale_scale = noise_scale
        masks_3level = []
        if dn_mask_noise_scale_shift > 0:
            masks = torch.cat(
                [F.interpolate(targets[i]['masks'].unsqueeze(1), size=size_list[-1], mode="nearest")
                 for i in range(len(targets))]).flatten(0, 1)
            masks = masks.repeat(scalar, 1, 1)
            # import ipdb; ipdb.set_trace()
            diff = torch.zeros_like(knwon_boxes_expand)
            diff[..., :2] = knwon_boxes_expand[..., 2:] / 2* dn_mask_noise_scale_shift
            diff[..., 2:] = knwon_boxes_expand[..., 2:]*dn_mask_noise_scale_scale
            delta_masks = torch.mul((torch.rand_like(knwon_boxes_expand) * 2 - 1.0), diff)
            delta_masks *= size_list[-1][-1]
            is_scale=torch.rand_like(knwon_boxes_expand[...,0])
            new_masks = []
            scale_size=(torch.tensor(size_list[-1]).float()*(1+dn_mask_noise_scale_scale)).long()+1
            delta_center=(torch.tensor(size_list[-1])-scale_size).to(knwon_boxes_expand)*knwon_boxes_expand[...,:2]
            scale_size = scale_size.tolist()
            for mask, delta_mask,sc,dc in zip(masks, delta_masks,is_scale,delta_center):
                # x, y = torch.where(mask<0.5)
                if sc>self.dn_ratio:
                    mask_scale=F.interpolate(mask[None][None],scale_size, mode="nearest")[0][0]
                    x_, y_ = torch.where(mask_scale > 0.5)
                    x_+=dc[0].long()
                    y_+=dc[1].long()
                else:
                    x_, y_ = torch.where(mask > 0.5)
                if shift_dn:
                    delta_x = delta_mask[0]
                    delta_y = delta_mask[1]
                    x_ = x_ + delta_x
                    y_ = y_ + delta_y
                x_ = x_.clamp(min=0, max=size_list[-1][-2] - 1)
                y_ = y_.clamp(min=0, max=size_list[-1][-1] - 1)
                mask = torch.ones_like(mask,dtype=torch.bool)
                mask[x_.long(), y_.long()] = False

                new_masks.append(mask)
            new_masks = torch.stack(new_masks)
            masks = new_masks.flatten(1)

            masks_3level.append(masks)
            # import ipdb; ipdb.set_trace()
            masks = (F.interpolate(new_masks.unsqueeze(1).float(), size=size_list[-2], mode="nearest") > 0.5).flatten(1)
            masks_3level.append(masks)

            masks = (F.interpolate(new_masks.unsqueeze(1).float(), size=size_list[0], mode="nearest") > 0.5).flatten(1)
            masks_3level.append(masks)

            # knwon_boxes_expand = knwon_boxes_expand.clamp(min=0.0, max=1.0)

        labels = torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(scalar, 1).view(-1)

        dn_label_noise_ratio = self.dn_label_noise_ratio
        knwon_labels_expand = known_labels.clone()
        if dn_label_noise_ratio > 0:
            prob = torch.rand_like(knwon_labels_expand.float())
            chosen_indice = prob < dn_label_noise_ratio
            new_label = torch.randint_like(knwon_labels_expand[chosen_indice], 0,
                                           self.num_classes)  # randomly put a new one here
            # gt_labels_expand.scatter_(0, chosen_indice, new_label)
            knwon_labels_expand[chosen_indice] = new_label

        known_labels = self.label_enc(knwon_labels_expand)
        # noised_known_features = known_labels

        # known_features = torch.cat([(m * mask_features[i]).flatten(-2).mean(-1) for i, m in enumerate(masks_)])
        # known_features = known_features.repeat(scalar, 1).detach()

        noised_known_features = known_labels


        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')

        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_boxes])  # [1,2, 1,2,3]
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(scalar)]).long().cuda()

        # for i in range(bs):
        padding[(known_bid, map_known_indices)] = noised_known_features

        for i, padding_mask in enumerate(padding_mask_3level):
            padding_mask_3level[i][(known_bid, map_known_indices)] = masks_3level[2-i]
            padding_mask_3level[i] = padding_mask_3level[i].unsqueeze(1).repeat([1, 8, 1, 1])

        res = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        res = torch.cat([padding.transpose(0, 1), res], dim=0)
        ###################

        outputs_class, outputs_mask, attn_mask_ = self.forward_prediction_heads(res, mask_features,
                                                                                attn_mask_target_size=size_list[0])
        attn_mask_ = attn_mask_.reshape([bs, 8, -1, attn_mask_.shape[-1]])
        attn_mask_[:, :, :-self.num_queries] = padding_mask_3level[0]
        attn_mask_ = attn_mask_.flatten(0, 1)
        #####################

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return None, res, attn_mask, dn_args_, outputs_class, outputs_mask, attn_mask_, padding_mask_3level
        # pass

    def prepare_for_dn_v8(self, mask_features, dn_args, size_list):
        targets, scalar, noise_scale = dn_args['tgt'], dn_args['scalar'], dn_args['noise_scale']
        boxes = [targets[i]['boxes'] for i in range(len(targets))]
        num_boxes = [len(b) for b in boxes]
        single_pad = max_num = max(num_boxes)
        if max_num == 0 or scalar == 0:
            return None
        if scalar >= 100:
            scalar = scalar // max_num
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size
        bs = len(boxes)

        padding = torch.zeros([bs, pad_size, self.query_feat.weight.shape[-1]]).cuda()
        padding_mask_3level = []
        for i in range(len(size_list)):
            padding_mask = torch.ones([bs, pad_size, size_list[i][0] * size_list[i][1]]).cuda().bool()
            padding_mask_3level.append(padding_mask)

        boxes = torch.cat([t['boxes'] for t in targets])  # x, y, x, y
        knwon_boxes_expand = boxes.repeat(scalar, 1)

        # knwon_labels_expand = known_labels.clone()
        dn_mask_noise_scale_shift = noise_scale
        dn_mask_noise_scale_scale = noise_scale
        masks_3level = []
        if dn_mask_noise_scale_shift > 0:
            masks = torch.cat(
                [F.interpolate(targets[i]['masks'].float().unsqueeze(1), size=size_list[-1], mode="nearest")
                 for i in range(len(targets))]).flatten(0, 1)
            masks = masks.repeat(scalar, 1, 1)
            edge_order=torch.rand_like(knwon_boxes_expand[...,0])
            new_masks = []
            masks = masks < 0.5
            areas = torch.clamp((~masks).sum(dim=[1,2]).float()*noise_scale,min=1.0)
            for mask,eo,area in zip(masks,edge_order,areas):
                max_short_edges=torch.sqrt(area).long()
                short_=random.randint(1,max_short_edges.item())
                long_=int(area/short_)
                if eo < 0.5:
                    patch_h=short_
                    patch_w=min(long_,mask.shape[1])
                else:
                    patch_h = min(long_,mask.shape[0])
                    patch_w = short_
                x0=random.randint(0,mask.shape[0]-patch_h)
                y0=random.randint(0,mask.shape[1]-patch_w)
                mask[x0:x0+patch_h,y0:y0+patch_w]=False

                new_masks.append(mask)
            new_masks = torch.stack(new_masks)
            masks = new_masks.flatten(1)

            masks_3level.append(masks)
            # import ipdb; ipdb.set_trace()
            masks = (F.interpolate(new_masks.unsqueeze(1).float(), size=size_list[-2], mode="nearest") > 0.5).flatten(1)
            masks_3level.append(masks)

            masks = (F.interpolate(new_masks.unsqueeze(1).float(), size=size_list[0], mode="nearest") > 0.5).flatten(1)
            masks_3level.append(masks)

            # knwon_boxes_expand = knwon_boxes_expand.clamp(min=0.0, max=1.0)

        labels = torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(scalar, 1).view(-1)

        dn_label_noise_ratio = self.dn_label_noise_ratio
        knwon_labels_expand = known_labels.clone()
        if dn_label_noise_ratio > 0:
            prob = torch.rand_like(knwon_labels_expand.float())
            chosen_indice = prob < dn_label_noise_ratio
            new_label = torch.randint_like(knwon_labels_expand[chosen_indice], 0,
                                           self.num_classes)  # randomly put a new one here
            # gt_labels_expand.scatter_(0, chosen_indice, new_label)
            knwon_labels_expand[chosen_indice] = new_label

        known_labels = self.label_enc(knwon_labels_expand)
        # noised_known_features = known_labels

        # known_features = torch.cat([(m * mask_features[i]).flatten(-2).mean(-1) for i, m in enumerate(masks_)])
        # known_features = known_features.repeat(scalar, 1).detach()
        # if dn_label_noise_ratio > 0:
        noised_known_features = known_labels
        # else:
        #     dn_delta = (torch.rand_like(known_labels) * 2 - 1.0) * noise_scale * known_labels
        #     noised_known_features = known_labels + dn_delta

        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')

        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_boxes])  # [1,2, 1,2,3]
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(scalar)]).long().cuda()

        # for i in range(bs):
        padding[(known_bid, map_known_indices)] = noised_known_features

        for i, padding_mask in enumerate(padding_mask_3level):
            padding_mask_3level[i][(known_bid, map_known_indices)] = masks_3level[2-i]
            padding_mask_3level[i] = padding_mask_3level[i].unsqueeze(1).repeat([1, 8, 1, 1])

        res = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        res = torch.cat([padding.transpose(0, 1), res], dim=0)
        ###################

        outputs_class, outputs_mask, attn_mask_ = self.forward_prediction_heads(res, mask_features,
                                                                                attn_mask_target_size=size_list[0])
        attn_mask_ = attn_mask_.reshape([bs, 8, -1, attn_mask_.shape[-1]])
        attn_mask_[:, :, :-self.num_queries] = padding_mask_3level[0]
        attn_mask_ = attn_mask_.flatten(0, 1)
        #####################

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return None, res, attn_mask, dn_args_, outputs_class, outputs_mask, attn_mask_, padding_mask_3level
        # pass

    def prepare_for_dn_v9(self, mask_features, dn_args, size_list):
        targets, scalar, noise_scale = dn_args['tgt'], dn_args['scalar'], dn_args['noise_scale']
        boxes = [targets[i]['boxes'] for i in range(len(targets))]
        num_boxes = [len(b) for b in boxes]
        single_pad = max_num = max(num_boxes)
        if max_num == 0 or scalar == 0:
            return None
        if scalar >= 100:
            scalar = scalar // max_num
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size
        bs = len(boxes)

        padding = torch.zeros([bs, pad_size, self.query_feat.weight.shape[-1]]).cuda()
        padding_mask_3level = []
        for i in range(len(size_list)):
            padding_mask = torch.ones([bs, pad_size, size_list[i][0] * size_list[i][1]]).cuda().bool()
            padding_mask_3level.append(padding_mask)

        boxes = torch.cat([t['boxes'] for t in targets])  # x, y, x, y
        knwon_boxes_expand = boxes.repeat(scalar, 1)

        # knwon_labels_expand = known_labels.clone()


        masks_3level = []
        if noise_scale > 0:
            masks = torch.cat(
                [F.interpolate(targets[i]['masks'].float().unsqueeze(1), size=size_list[-1], mode="nearest")
                 for i in range(len(targets))]).flatten(0, 1)
            masks = masks.repeat(scalar, 1, 1)
            edge_order = torch.rand_like(knwon_boxes_expand[..., 0])
            new_masks = []
            masks = masks < 0.5
            noise= torch.rand_like(masks.float())
            noise=noise<noise_scale
            new_masks=torch.logical_or(masks,noise)

            masks_3level.append(new_masks.flatten(1))
            # import ipdb; ipdb.set_trace()
            masks = (F.interpolate(masks.unsqueeze(1).float(), size=size_list[-2],
                                   mode="nearest") > 0.5).flatten(0, 1)

            noise = torch.rand_like(masks.float())
            noise = noise < noise_scale
            new_masks = torch.logical_or(masks, noise)

            masks_3level.append(new_masks.flatten(1))

            masks = (F.interpolate(masks.unsqueeze(1).float(), size=size_list[0],
                                   mode="nearest") > 0.5).flatten(0, 1)
            noise = torch.rand_like(masks.float())
            noise = noise < noise_scale
            new_masks = torch.logical_or(masks, noise)
            masks_3level.append(new_masks.flatten(1))

            # knwon_boxes_expand = knwon_boxes_expand.clamp(min=0.0, max=1.0)

        labels = torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(scalar, 1).view(-1)

        dn_label_noise_ratio = self.dn_label_noise_ratio
        knwon_labels_expand = known_labels.clone()
        if dn_label_noise_ratio > 0:
            prob = torch.rand_like(knwon_labels_expand.float())
            chosen_indice = prob < dn_label_noise_ratio
            new_label = torch.randint_like(knwon_labels_expand[chosen_indice], 0,
                                           self.num_classes)  # randomly put a new one here
            # gt_labels_expand.scatter_(0, chosen_indice, new_label)
            knwon_labels_expand[chosen_indice] = new_label

        known_labels = self.label_enc(knwon_labels_expand)
        # noised_known_features = known_labels

        # known_features = torch.cat([(m * mask_features[i]).flatten(-2).mean(-1) for i, m in enumerate(masks_)])
        # known_features = known_features.repeat(scalar, 1).detach()
        # if dn_label_noise_ratio > 0:
        noised_known_features = known_labels
        # else:
        #     dn_delta = (torch.rand_like(known_labels) * 2 - 1.0) * noise_scale * known_labels
        #     noised_known_features = known_labels + dn_delta

        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')

        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_boxes])  # [1,2, 1,2,3]
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(scalar)]).long().cuda()

        # for i in range(bs):
        padding[(known_bid, map_known_indices)] = noised_known_features

        for i, padding_mask in enumerate(padding_mask_3level):
            padding_mask_3level[i][(known_bid, map_known_indices)] = masks_3level[2 - i]
            padding_mask_3level[i] = padding_mask_3level[i].unsqueeze(1).repeat([1, 8, 1, 1])

        res = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        res = torch.cat([padding.transpose(0, 1), res], dim=0)
        ###################

        outputs_class, outputs_mask, attn_mask_ = self.forward_prediction_heads(res, mask_features,
                                                                                attn_mask_target_size=size_list[0])
        attn_mask_ = attn_mask_.reshape([bs, 8, -1, attn_mask_.shape[-1]])
        attn_mask_[:, :, :-self.num_queries] = padding_mask_3level[0]
        attn_mask_ = attn_mask_.flatten(0, 1)
        #####################

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return None, res, attn_mask, dn_args_, outputs_class, outputs_mask, attn_mask_, padding_mask_3level
        # pass
        # pass

    def gen_mask_dn(self,dn_args,size_list,known_bid,map_known_indices):
        targets, scalar, noise_scale = dn_args['tgt'], dn_args['scalar'], dn_args['noise_scale']
        boxes = [targets[i]['boxes'] for i in range(len(targets))]
        num_boxes = [len(b) for b in boxes]
        max_num = max(num_boxes)
        if max_num == 0 or scalar == 0:
            return None
        if scalar >= 100:
            scalar = scalar // max_num
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size
        bs = len(boxes)

        padding_mask = torch.ones([bs, pad_size, size_list[0] * size_list[1]]).cuda().bool()
        masks = torch.cat(
            [F.interpolate(targets[i]['masks'].float().unsqueeze(1), size=size_list, mode="area").flatten(1) <= 1e-8
             for i in range(len(targets)) if len(targets[i]['masks']) > 0]).repeat(scalar, 1)
        if self.head_dn:
            masks = masks[:, None].repeat(1, 8, 1)
            areas = (~masks).sum(2)
            noise_ratio = areas * noise_scale / (size_list[0] * size_list[1])
            delta_mask = torch.rand_like(masks, dtype=torch.float) < noise_ratio[:, :, None]
            masks = torch.logical_xor(masks, delta_mask)  # [q,h,h*w]
        else:
            areas = (~masks).sum(1)
            noise_ratio = areas * noise_scale / (size_list[0] * size_list[1])
            delta_mask = torch.rand_like(masks, dtype=torch.float) < noise_ratio[:, None]
            masks = torch.logical_xor(masks, delta_mask)

        if self.head_dn:
            padding_mask = padding_mask.unsqueeze(2).repeat([1, 1, 8, 1])  # [bs,q,h,h*w]
            padding_mask[(known_bid, map_known_indices)] = masks
            padding_mask = padding_mask.transpose(1, 2)
        else:
            padding_mask[(known_bid, map_known_indices)] = masks
            padding_mask = padding_mask.unsqueeze(1).repeat([1, 8, 1, 1])
        return padding_mask

    def prepare_for_dn(self, mask_features, dn_args,size_list):
        targets, scalar, noise_scale = dn_args['tgt'], dn_args['scalar'], dn_args['noise_scale']
        boxes = [targets[i]['boxes'] for i in range(len(targets))]
        num_boxes = [len(b) for b in boxes]
        single_pad = max_num = max(num_boxes)
        scalar=1
        if max_num == 0 or scalar==0:
            return None
        if scalar >= 100:
            scalar = scalar // max_num
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size
        bs = len(boxes)

        padding = torch.zeros([bs, pad_size, self.query_feat.weight.shape[-1]]).cuda()
        padding_mask = torch.ones([bs, pad_size, size_list[0][0]*size_list[0][1]]).cuda().bool()
        # padding[]
        # gt_masks=torch.cat([])
        # import pdb;pdb.set_trace()
        #attn_mask[B * h, Q, HW]
        #masks [b*h,pad_size,hw]
        # masks_ = [F.interpolate(targets[i]['masks'].unsqueeze(1).float(), size=mask_features.shape[-2:], mode="bilinear")
        #          for i in range(len(targets))]
        masks = torch.cat([F.interpolate(targets[i]['masks'].unsqueeze(1), size=size_list[0], mode="nearest").flatten(1)<0.5
                 for i in range(len(targets))]).repeat(scalar, 1)

        labels=torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(scalar, 1).view(-1)
        known_labels = self.label_enc(known_labels)
        noised_known_features = known_labels

        # known_features = torch.cat([(m * mask_features[i]).flatten(-2).mean(-1) for i, m in enumerate(masks_)])
        # known_features = known_features.repeat(scalar, 1).detach()

        # dn_delta = (torch.rand_like(known_features) * 2 - 1.0) * noise_scale * known_features
        # noised_known_features = known_features + dn_delta

        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')

        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_boxes])  # [1,2, 1,2,3]
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(scalar)]).long().cuda()

        # for i in range(bs):
        padding[(known_bid, map_known_indices)] = noised_known_features
        padding_mask[(known_bid, map_known_indices)]=masks
        padding_mask=padding_mask.unsqueeze(1).repeat([1,8,1,1])

        res = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        res = torch.cat([padding.transpose(0, 1), res], dim=0)
        ###################

        outputs_class, outputs_mask, attn_mask_ = self.forward_prediction_heads(res, mask_features,attn_mask_target_size=size_list[0])
        attn_mask_=attn_mask_.view([bs,8,-1,attn_mask_.shape[-1]])
        attn_mask_[:,:,:-self.num_queries]=padding_mask
        attn_mask_=attn_mask_.flatten(0,1)
        #####################

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return None, res, attn_mask, dn_args_,outputs_class, outputs_mask, attn_mask_
        # pass

    def postprocess_for_dn(self, predictions_class, predictions_mask):
        n_lys = len(predictions_class)
        dn_predictions_class, predictions_class = [predictions_class[i][:, :-self.num_queries] for i in range(n_lys)], \
                                                  [predictions_class[i][:, -self.num_queries:] for i in range(n_lys)]
        dn_predictions_mask, predictions_mask = [predictions_mask[i][:, :-self.num_queries] for i in range(n_lys)], \
                                                [predictions_mask[i][:, -self.num_queries:] for i in range(n_lys)]
        return predictions_class, predictions_mask, dn_predictions_class, dn_predictions_mask
        # pass

    def forward(self, x, mask_features, mask=None, dn_args=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        if dn_args is not None:
            if self.dn_mode=='base':
                res= self.prepare_for_dn(mask_features, dn_args,size_list)
            elif self.dn_mode=='lb':
                res= self.prepare_for_dn_v2(mask_features, dn_args,size_list)
            elif self.dn_mode=='mask':
                res=self.prepare_for_dn_v3(mask_features, dn_args,size_list)
            elif self.dn_mode=='points':
                res= self.prepare_for_dn_v5(mask_features, dn_args,size_list)
            # elif self.dn_mode=='multi_ly_lb':
            #     res= self.prepare_for_dn_v5(mask_features, dn_args,size_list)
            elif self.dn_mode=='shift':
                res= self.prepare_for_dn_v6(mask_features, dn_args,size_list)
            elif self.dn_mode=='scale':
                res= self.prepare_for_dn_v7(mask_features, dn_args,size_list)
            elif self.dn_mode=='shift_scale':
                res= self.prepare_for_dn_v7(mask_features, dn_args,size_list,shift_dn=True)
            elif self.dn_mode=='patch':
                res= self.prepare_for_dn_v8(mask_features, dn_args,size_list)
            elif self.dn_mode=='points_MAE':
                res= self.prepare_for_dn_v9(mask_features, dn_args,size_list)
            else:
                res= None
            if res is None:
                query_embed, output, tgt_mask, outputs_class, outputs_mask, attn_mask = \
                    self.prepare_for_normal(bs, mask_features, size_list)
            else:
                if self.dn_mode!='points':
                    query_embed, output, tgt_mask, dn_args_, outputs_class, outputs_mask, attn_mask,padding_mask_3level\
                        = res
                else:
                    query_embed, output, tgt_mask, dn_args_,outputs_class, outputs_mask, attn_mask=res
                    known_bid,map_known_indices=query_embed
                    query_embed=None
        else:
            query_embed, output, tgt_mask, outputs_class, outputs_mask, attn_mask=\
                self.prepare_for_normal(bs,mask_features,size_list)
        # import pdb;pdb.set_trace()

        predictions_class = []
        predictions_mask = []
        dn_predictions_class = []
        dn_predictions_mask = []

        # prediction heads on learnable query features
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # import pdb;pdb.set_trace()
            # Deal with extreme cases when all elements are masked
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # import pdb;
            # pdb.set_trace()
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=tgt_mask,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )
            # attn_mask:[B*h, Q, HW]

            if self.all_lys:
                flag=True
            else:
                flag=i<3
            level = (i + 1) % self.num_feature_levels

            if dn_args is not None and tgt_mask is not None and flag:
                if self.dn_mode=="points":
                    outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                                           attn_mask_target_size=size_list[level])
                    padding_mask=self.gen_mask_dn(dn_args,size_list=size_list[(i + 1) % self.num_feature_levels],known_bid=known_bid,map_known_indices=map_known_indices)
                    attn_mask = attn_mask.view([bs, 8, -1, attn_mask.shape[-1]])
                    attn_mask[:, :, :-self.num_queries] = padding_mask
                    attn_mask = attn_mask.flatten(0, 1)
                else:
                    outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads_dn(output, mask_features,
                                                                                           attn_mask_target_size=size_list[level]
                                                                                           ,padding_mask=padding_mask_3level[level])
            else:
                outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                                       attn_mask_target_size=size_list[
                                                                                           level]
                                                                                       )

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1
        if not (tgt_mask is None):
            predictions_class, predictions_mask, dn_predictions_class, dn_predictions_mask = \
                self.postprocess_for_dn(predictions_class, predictions_mask)

            dn_out = {
                'pred_logits': dn_predictions_class[-1],
                'pred_masks': dn_predictions_mask[-1],
                'aux_outputs': self._set_aux_loss(
                    dn_predictions_class if self.mask_classification else None, dn_predictions_mask
                ),
                'dn_args': dn_args_

            }
        else:
            dn_out = None
            predictions_class[-1]+=self.label_enc.weight[0,0]*0.0

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
            'dn_out': dn_out
        }

        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        # import pdb;pdb.set_trace()
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # import pdb;pdb.set_trace()
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                         1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    def forward_prediction_heads_dn(self, output, mask_features, attn_mask_target_size,padding_mask=None):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        # import pdb;pdb.set_trace()
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # import pdb;pdb.set_trace()
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        if padding_mask is not None:
            attn_mask = attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn_mask[:,:,:-self.num_queries] = padding_mask
            attn_mask= (attn_mask.flatten(0,1) < 0.5).bool()
        else:
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()


        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoderMaskDNLYSmooth(nn.Module):
    _version = 2

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
            self,
            in_channels,
            mask_classification=True,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,
            mask_dim: int,
            enforce_input_project: bool,
            dn_mode="base",
            head_dn=False,
            all_lys=False,
            dn_ratio=0.5,
            dn_label_noise_ratio=-1.,
            last_q_ratio=0.0,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        self.head_dn=head_dn
        self.dn_ratio=dn_ratio
        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.dn_label_noise_ratio=dn_label_noise_ratio
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_classes=num_classes
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.dn_mode=dn_mode
        self.last_q_ratio=last_q_ratio

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.matching_dict=dict()
        self.label_enc = nn.Embedding(num_classes, hidden_dim)
        self.all_lys=all_lys

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["dn_mode"] = cfg.MODEL.MASK_FORMER.DN_MODE
        ret["head_dn"] = cfg.MODEL.MASK_FORMER.HEAD_DN
        ret["all_lys"] = cfg.MODEL.MASK_FORMER.ALL_LY_DN
        ret["dn_ratio"] = cfg.MODEL.MASK_FORMER.DN_RATIO
        ret["last_q_ratio"] = cfg.MODEL.MASK_FORMER.LAST_Q_RATIO
        ret["dn_label_noise_ratio"] = cfg.MODEL.MASK_FORMER.LB_NOISE_RATIO

        return ret

    def prepare_for_normal(self,bs,mask_features,size_list):
        query_embed = None
        tgt_mask = None
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        outputs_class, outputs_mask, attn_mask,_ = self.forward_prediction_heads(output, mask_features,
                                                                               attn_mask_target_size=size_list[0])
        return query_embed,output,tgt_mask,outputs_class, outputs_mask, attn_mask

    def prepare_for_dn_v2(self, mask_features, dn_args,size_list):
        targets, scalar, noise_scale = dn_args['tgt'], dn_args['scalar'], dn_args['noise_scale']
        boxes = [targets[i]['boxes'] for i in range(len(targets))]
        num_boxes = [len(b) for b in boxes]
        single_pad = max_num = max(num_boxes)
        if max_num == 0 or scalar==0:
            return None
        if scalar >= 100:
            scalar = scalar // max_num
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size
        bs = len(boxes)

        padding = torch.zeros([bs, pad_size, self.query_feat.weight.shape[-1]]).cuda()
        padding_mask = torch.ones([bs, pad_size, size_list[0][0]*size_list[0][1]]).cuda().bool()
        # padding[]
        # gt_masks=torch.cat([])
        # import pdb;pdb.set_trace()
        #attn_mask[B * h, Q, HW]
        #masks [b*h,pad_size,hw]
        # masks_ = [F.interpolate(targets[i]['masks'].unsqueeze(1).float(), size=mask_features.shape[-2:], mode="bilinear")
        #          for i in range(len(targets))]
        masks = torch.cat([F.interpolate(targets[i]['masks'].unsqueeze(1), size=size_list[0], mode="nearest").flatten(1)<0.5
                 for i in range(len(targets))]).repeat(scalar, 1)

        labels=torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(scalar, 1).view(-1)
        known_labels = self.label_enc(known_labels)
        # noised_known_features = known_labels

        # known_features = torch.cat([(m * mask_features[i]).flatten(-2).mean(-1) for i, m in enumerate(masks_)])
        # known_features = known_features.repeat(scalar, 1).detach()

        dn_delta = (torch.rand_like(known_labels) * 2 - 1.0) * noise_scale * known_labels
        noised_known_features = known_labels + dn_delta

        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')

        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_boxes])  # [1,2, 1,2,3]
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(scalar)]).long().cuda()

        # for i in range(bs):
        padding[(known_bid, map_known_indices)] = noised_known_features
        padding_mask[(known_bid, map_known_indices)]=masks
        padding_mask=padding_mask.unsqueeze(1).repeat([1,8,1,1])

        res = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        res = torch.cat([padding.transpose(0, 1), res], dim=0)
        ###################

        outputs_class, outputs_mask, attn_mask_ = self.forward_prediction_heads(res, mask_features,attn_mask_target_size=size_list[0])
        attn_mask_=attn_mask_.view([bs,8,-1,attn_mask_.shape[-1]])
        attn_mask_[:,:,:-self.num_queries]=padding_mask
        attn_mask_=attn_mask_.flatten(0,1)
        #####################

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return None, res, attn_mask, dn_args_,outputs_class, outputs_mask, attn_mask_
        # pass

    def prepare_for_dn_v3(self, mask_features, dn_args,size_list):
        targets, scalar, noise_scale = dn_args['tgt'], dn_args['scalar'], dn_args['noise_scale']
        boxes = [targets[i]['boxes'] for i in range(len(targets))]
        num_boxes = [len(b) for b in boxes]
        single_pad = max_num = max(num_boxes)
        if max_num == 0 or scalar==0:
            return None
        if scalar >= 100:
            scalar = scalar // max_num
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size
        bs = len(boxes)

        padding = torch.zeros([bs, pad_size, self.query_feat.weight.shape[-1]]).cuda()
        padding_mask = torch.ones([bs, pad_size, size_list[0][0]*size_list[0][1]]).cuda().bool()

        masks = torch.cat([F.interpolate(targets[i]['masks'].float().unsqueeze(1), size=size_list[0], mode="area").flatten(1)<=1e-8
                 for i in range(len(targets)) if len(targets[i]['masks'])>0]).repeat(scalar, 1)
        head_dn=self.head_dn
        if head_dn:
            masks = masks[:, None].repeat(1, 8, 1)
            areas = (~masks).sum(2)
            noise_ratio = areas * noise_scale / (size_list[0][0] * size_list[0][1])
            delta_mask = torch.rand_like(masks, dtype=torch.float) < noise_ratio[:,:, None]
            masks = torch.logical_xor(masks, delta_mask) #[q,h,h*w]
        else:
            areas= (~masks).sum(1)
            noise_ratio=areas*noise_scale/(size_list[0][0]*size_list[0][1])
            delta_mask=torch.rand_like(masks,dtype=torch.float)<noise_ratio[:,None]
            masks=torch.logical_xor(masks,delta_mask)


        # torch.sum(dim=)

        labels=torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(scalar, 1).view(-1)
        known_labels = self.label_enc(known_labels)
        noised_known_features = known_labels

        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')

        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_boxes])  # [1,2, 1,2,3]
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(scalar)]).long().cuda()

        # for i in range(bs):
        padding[(known_bid, map_known_indices)] = noised_known_features
        if head_dn:
            padding_mask=padding_mask.unsqueeze(2).repeat([1,1,8,1]) #[bs,q,h,h*w]
            padding_mask[(known_bid, map_known_indices)] = masks
            padding_mask=padding_mask.transpose(1,2)
        else:
            padding_mask[(known_bid, map_known_indices)]=masks
            padding_mask=padding_mask.unsqueeze(1).repeat([1,8,1,1])

        res = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        res = torch.cat([padding.transpose(0, 1), res], dim=0)
        ###################

        outputs_class, outputs_mask, attn_mask_ = self.forward_prediction_heads(res, mask_features,attn_mask_target_size=size_list[0])
        attn_mask_=attn_mask_.view([bs,8,-1,attn_mask_.shape[-1]])
        attn_mask_[:,:,:-self.num_queries]=padding_mask
        attn_mask_=attn_mask_.flatten(0,1)
        #####################

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return None, res, attn_mask, dn_args_,outputs_class, outputs_mask, attn_mask_
        # pass

    def prepare_for_dn_v4(self, mask_features, dn_args,size_list):
        targets, scalar, noise_scale = dn_args['tgt'], dn_args['scalar'], dn_args['noise_scale']
        boxes = [targets[i]['boxes'] for i in range(len(targets))]
        num_boxes = [len(b) for b in boxes]
        single_pad = max_num = max(num_boxes)
        if max_num == 0 or scalar==0:
            return None
        if scalar >= 100:
            scalar = scalar // max_num
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size
        bs = len(boxes)

        padding = torch.zeros([bs, pad_size, self.query_feat.weight.shape[-1]]).cuda()
        padding_mask = torch.ones([bs, pad_size, size_list[0][0]*size_list[0][1]]).cuda().bool()
        masks = torch.cat([F.interpolate(targets[i]['masks'].float().unsqueeze(1), size=size_list[0], mode="area").flatten(1)<=1e-8
                 for i in range(len(targets)) if len(targets[i]['masks'])>0]).repeat(scalar, 1)
        if self.head_dn:
            masks = masks[:, None].repeat(1, 8, 1)
            areas = (~masks).sum(2)
            noise_ratio = areas * noise_scale / (size_list[0][0] * size_list[0][1])
            delta_mask = torch.rand_like(masks, dtype=torch.float) < noise_ratio[:,:, None]
            masks = torch.logical_xor(masks, delta_mask) #[q,h,h*w]
        else:
            areas= (~masks).sum(1)
            noise_ratio=areas*noise_scale/(size_list[0][0]*size_list[0][1])
            delta_mask=torch.rand_like(masks,dtype=torch.float)<noise_ratio[:,None]
            masks=torch.logical_xor(masks,delta_mask)


        # torch.sum(dim=)

        labels=torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(scalar, 1).view(-1)
        known_labels = self.label_enc(known_labels)
        noised_known_features = known_labels


        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')

        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_boxes])  # [1,2, 1,2,3]
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(scalar)]).long().cuda()

        # for i in range(bs):
        padding[(known_bid, map_known_indices)] = noised_known_features
        if self.head_dn:
            padding_mask=padding_mask.unsqueeze(2).repeat([1,1,8,1]) #[bs,q,h,h*w]
            padding_mask[(known_bid, map_known_indices)] = masks
            padding_mask=padding_mask.transpose(1,2)
        else:
            padding_mask[(known_bid, map_known_indices)]=masks
            padding_mask=padding_mask.unsqueeze(1).repeat([1,8,1,1])

        res = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        res = torch.cat([padding.transpose(0, 1), res], dim=0)
        ###################

        outputs_class, outputs_mask, attn_mask_ = self.forward_prediction_heads(res, mask_features,attn_mask_target_size=size_list[0])
        attn_mask_=attn_mask_.view([bs,8,-1,attn_mask_.shape[-1]])
        attn_mask_[:,:,:-self.num_queries]=padding_mask
        attn_mask_=attn_mask_.flatten(0,1)
        #####################

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return (known_bid,map_known_indices), res, attn_mask, dn_args_,outputs_class, outputs_mask, attn_mask_
        # pass

    def prepare_for_dn_v5(self, mask_features, dn_args,size_list):
        targets, scalar, noise_scale = dn_args['tgt'], dn_args['scalar'], dn_args['noise_scale']
        boxes = [targets[i]['boxes'] for i in range(len(targets))]
        num_boxes = [len(b) for b in boxes]
        single_pad = max_num = max(num_boxes)
        if max_num == 0 or scalar==0:
            return None
        if scalar >= 100:
            scalar = scalar // max_num
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size
        bs = len(boxes)

        padding = torch.zeros([bs, pad_size, self.query_feat.weight.shape[-1]]).cuda()
        padding_mask = torch.ones([bs, pad_size, size_list[0][0]*size_list[0][1]]).cuda().bool()
        masks = torch.cat([F.interpolate(targets[i]['masks'].float().unsqueeze(1), size=size_list[0], mode="area").flatten(1)<=1e-8
                 for i in range(len(targets)) if len(targets[i]['masks'])>0]).repeat(scalar, 1)
        if self.head_dn:
            masks = masks[:, None].repeat(1, 8, 1)
            areas = (~masks).sum(2)
            noise_ratio = areas * noise_scale / (size_list[0][0] * size_list[0][1])
            delta_mask = torch.rand_like(masks, dtype=torch.float) < noise_ratio[:,:, None]
            masks = torch.logical_xor(masks, delta_mask) #[q,h,h*w]
        else:
            areas= (~masks).sum(1)
            noise_ratio=areas*noise_scale/(size_list[0][0]*size_list[0][1])
            delta_mask=torch.rand_like(masks,dtype=torch.float)<noise_ratio[:,None]
            masks=torch.logical_xor(masks,delta_mask)


        # torch.sum(dim=)

        labels=torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(scalar, 1).view(-1)
        # known_labels = self.label_enc(known_labels)

        dn_label_noise_ratio = self.dn_label_noise_ratio
        knwon_labels_expand = known_labels.clone()
        if dn_label_noise_ratio > 0:
            prob = torch.rand_like(knwon_labels_expand.float())
            chosen_indice = prob < dn_label_noise_ratio
            new_label = torch.randint_like(knwon_labels_expand[chosen_indice], 0,
                                           self.num_classes)  # randomly put a new one here
            # gt_labels_expand.scatter_(0, chosen_indice, new_label)
            knwon_labels_expand[chosen_indice] = new_label

        known_labels = self.label_enc(knwon_labels_expand)
        # noised_known_features = known_labels
        noised_known_features = known_labels


        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')

        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_boxes])  # [1,2, 1,2,3]
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(scalar)]).long().cuda()

        # for i in range(bs):
        padding[(known_bid, map_known_indices)] = noised_known_features
        if self.head_dn:
            padding_mask=padding_mask.unsqueeze(2).repeat([1,1,8,1]) #[bs,q,h,h*w]
            padding_mask[(known_bid, map_known_indices)] = masks
            padding_mask=padding_mask.transpose(1,2)
        else:
            padding_mask[(known_bid, map_known_indices)]=masks
            padding_mask=padding_mask.unsqueeze(1).repeat([1,8,1,1])

        res = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        res = torch.cat([padding.transpose(0, 1), res], dim=0)
        ###################

        outputs_class, outputs_mask, attn_mask_,_ = self.forward_prediction_heads(res, mask_features,attn_mask_target_size=size_list[0])
        attn_mask_=attn_mask_.view([bs,8,-1,attn_mask_.shape[-1]])
        attn_mask_[:,:,:-self.num_queries]=padding_mask
        attn_mask_=attn_mask_.flatten(0,1)
        #####################

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return (known_bid,map_known_indices), res, attn_mask, dn_args_,outputs_class, outputs_mask, attn_mask_
        # pass

    def prepare_for_dn_v6(self, mask_features, dn_args, size_list):
        targets, scalar, noise_scale = dn_args['tgt'], dn_args['scalar'], dn_args['noise_scale']
        boxes = [targets[i]['boxes'] for i in range(len(targets))]
        num_boxes = [len(b) for b in boxes]
        single_pad = max_num = max(num_boxes)
        if max_num == 0 or scalar == 0:
            return None
        if scalar >= 100:
            scalar = scalar // max_num
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size
        bs = len(boxes)

        padding = torch.zeros([bs, pad_size, self.query_feat.weight.shape[-1]]).cuda()
        padding_mask_3level = []
        for i in range(len(size_list)):
            padding_mask = torch.ones([bs, pad_size, size_list[i][0] * size_list[i][1]]).cuda().bool()
            padding_mask_3level.append(padding_mask)

        # padding[]
        # gt_masks=torch.cat([])
        # import pdb;pdb.set_trace()
        # attn_mask[B * h, Q, HW]
        # masks [b*h,pad_size,hw]
        # masks_ = [F.interpolate(targets[i]['masks'].unsqueeze(1).float(), size=mask_features.shape[-2:], mode="bilinear")
        #          for i in range(len(targets))]
        # True means no object
        # import ipdb; ipdb.set_trace()
        # masks0 = torch.cat([F.interpolate(targets[i]['masks'].unsqueeze(1), size=size_list[0], mode="nearest").flatten(1)<0.5
        #          for i in range(len(targets))]).repeat(scalar, 1)
        boxes = torch.cat([t['boxes'] for t in targets])  # x, y, x, y
        knwon_boxes_expand = boxes.repeat(scalar, 1)

        # knwon_labels_expand = known_labels.clone()
        dn_mask_noise_scale = noise_scale
        masks_3level = []
        if dn_mask_noise_scale > 0:
            masks = torch.cat(
                [F.interpolate(targets[i]['masks'].unsqueeze(1), size=size_list[-1], mode="nearest") < 0.5
                 for i in range(len(targets))]).flatten(0, 1)
            masks = masks.repeat(scalar, 1, 1)
            # import ipdb; ipdb.set_trace()
            diff = torch.zeros_like(knwon_boxes_expand)
            diff[..., :2] = knwon_boxes_expand[..., 2:] / 2
            diff[..., 2:] = knwon_boxes_expand[..., 2:]
            delta_masks = torch.mul((torch.rand_like(knwon_boxes_expand) * 2 - 1.0), diff) * dn_mask_noise_scale
            delta_masks *= size_list[-1][-1]
            new_masks = []
            for mask, delta_mask in zip(masks, delta_masks):
                x, y = torch.where(mask == False)
                delta_x = delta_mask[0]
                delta_y = delta_mask[1]
                # import ipdb; ipdb.set_trace()
                x = x + delta_x
                y = y + delta_y
                x = x.clamp(min=0, max=size_list[-1][-2] - 1)
                y = y.clamp(min=0, max=size_list[-1][-1] - 1)
                mask = torch.ones_like(mask)
                mask[x.long(), y.long()] = 0
                mask = mask > 0.5
                new_masks.append(mask)
            new_masks = torch.stack(new_masks)
            masks = new_masks.flatten(1)

            masks_3level.append(masks)
            # import ipdb; ipdb.set_trace()
            masks = (F.interpolate(new_masks.unsqueeze(1).float(), size=size_list[-2], mode="nearest") > 0.5).flatten(1)
            masks_3level.append(masks)

            masks = (F.interpolate(new_masks.unsqueeze(1).float(), size=size_list[0], mode="nearest") > 0.5).flatten(1)
            masks_3level.append(masks)

            # knwon_boxes_expand = knwon_boxes_expand.clamp(min=0.0, max=1.0)

        labels = torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(scalar, 1).view(-1)

        dn_label_noise_ratio = self.dn_label_noise_ratio
        knwon_labels_expand = known_labels.clone()
        if dn_label_noise_ratio > 0:
            prob = torch.rand_like(knwon_labels_expand.float())
            chosen_indice = prob < dn_label_noise_ratio
            new_label = torch.randint_like(knwon_labels_expand[chosen_indice], 0,
                                           self.num_classes)  # randomly put a new one here
            # gt_labels_expand.scatter_(0, chosen_indice, new_label)
            knwon_labels_expand[chosen_indice] = new_label

        known_labels = self.label_enc(knwon_labels_expand)
        # noised_known_features = known_labels

        # known_features = torch.cat([(m * mask_features[i]).flatten(-2).mean(-1) for i, m in enumerate(masks_)])
        # known_features = known_features.repeat(scalar, 1).detach()
        if dn_label_noise_ratio > 0:
            noised_known_features = known_labels
        else:
            dn_delta = (torch.rand_like(known_labels) * 2 - 1.0) * noise_scale * known_labels
            noised_known_features = known_labels + dn_delta

        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')

        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_boxes])  # [1,2, 1,2,3]
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(scalar)]).long().cuda()

        # for i in range(bs):
        padding[(known_bid, map_known_indices)] = noised_known_features

        for i, padding_mask in enumerate(padding_mask_3level):
            padding_mask_3level[i][(known_bid, map_known_indices)] = masks_3level[2-i]
            padding_mask_3level[i] = padding_mask_3level[i].unsqueeze(1).repeat([1, 8, 1, 1])

        res = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        res = torch.cat([padding.transpose(0, 1), res], dim=0)
        ###################

        outputs_class, outputs_mask, attn_mask_ = self.forward_prediction_heads(res, mask_features,
                                                                                attn_mask_target_size=size_list[0])
        attn_mask_ = attn_mask_.reshape([bs, 8, -1, attn_mask_.shape[-1]])
        attn_mask_[:, :, :-self.num_queries] = padding_mask_3level[0]
        attn_mask_ = attn_mask_.flatten(0, 1)
        #####################

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return None, res, attn_mask, dn_args_, outputs_class, outputs_mask, attn_mask_, padding_mask_3level
        # pass

    def prepare_for_dn_v7(self, mask_features, dn_args, size_list, shift_dn=False):
        targets, scalar, noise_scale = dn_args['tgt'], dn_args['scalar'], dn_args['noise_scale']
        boxes = [targets[i]['boxes'] for i in range(len(targets))]
        num_boxes = [len(b) for b in boxes]
        single_pad = max_num = max(num_boxes)
        if max_num == 0 or scalar == 0:
            return None
        if scalar >= 100:
            scalar = scalar // max_num
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size
        bs = len(boxes)

        padding = torch.zeros([bs, pad_size, self.query_feat.weight.shape[-1]]).cuda()
        padding_mask_3level = []
        for i in range(len(size_list)):
            padding_mask = torch.ones([bs, pad_size, size_list[i][0] * size_list[i][1]]).cuda().bool()
            padding_mask_3level.append(padding_mask)

        boxes = torch.cat([t['boxes'] for t in targets])  # x, y, x, y
        knwon_boxes_expand = boxes.repeat(scalar, 1)

        # knwon_labels_expand = known_labels.clone()
        dn_mask_noise_scale_shift = noise_scale
        dn_mask_noise_scale_scale = noise_scale
        masks_3level = []
        if dn_mask_noise_scale_shift > 0:
            masks = torch.cat(
                [F.interpolate(targets[i]['masks'].unsqueeze(1), size=size_list[-1], mode="nearest")
                 for i in range(len(targets))]).flatten(0, 1)
            masks = masks.repeat(scalar, 1, 1)
            # import ipdb; ipdb.set_trace()
            diff = torch.zeros_like(knwon_boxes_expand)
            diff[..., :2] = knwon_boxes_expand[..., 2:] / 2* dn_mask_noise_scale_shift
            diff[..., 2:] = knwon_boxes_expand[..., 2:]*dn_mask_noise_scale_scale
            delta_masks = torch.mul((torch.rand_like(knwon_boxes_expand) * 2 - 1.0), diff)
            delta_masks *= size_list[-1][-1]
            is_scale=torch.rand_like(knwon_boxes_expand[...,0])
            new_masks = []
            scale_size=(torch.tensor(size_list[-1]).float()*(1+dn_mask_noise_scale_scale)).long()+1
            delta_center=(torch.tensor(size_list[-1])-scale_size).to(knwon_boxes_expand)*knwon_boxes_expand[...,:2]
            scale_size = scale_size.tolist()
            for mask, delta_mask,sc,dc in zip(masks, delta_masks,is_scale,delta_center):
                # x, y = torch.where(mask<0.5)
                if sc>self.dn_ratio:
                    mask_scale=F.interpolate(mask[None][None],scale_size, mode="nearest")[0][0]
                    x_, y_ = torch.where(mask_scale > 0.5)
                    x_+=dc[0].long()
                    y_+=dc[1].long()
                else:
                    x_, y_ = torch.where(mask > 0.5)
                if shift_dn:
                    delta_x = delta_mask[0]
                    delta_y = delta_mask[1]
                    x_ = x_ + delta_x
                    y_ = y_ + delta_y
                x_ = x_.clamp(min=0, max=size_list[-1][-2] - 1)
                y_ = y_.clamp(min=0, max=size_list[-1][-1] - 1)
                mask = torch.ones_like(mask,dtype=torch.bool)
                mask[x_.long(), y_.long()] = False

                new_masks.append(mask)
            new_masks = torch.stack(new_masks)
            masks = new_masks.flatten(1)

            masks_3level.append(masks)
            # import ipdb; ipdb.set_trace()
            masks = (F.interpolate(new_masks.unsqueeze(1).float(), size=size_list[-2], mode="nearest") > 0.5).flatten(1)
            masks_3level.append(masks)

            masks = (F.interpolate(new_masks.unsqueeze(1).float(), size=size_list[0], mode="nearest") > 0.5).flatten(1)
            masks_3level.append(masks)

            # knwon_boxes_expand = knwon_boxes_expand.clamp(min=0.0, max=1.0)

        labels = torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(scalar, 1).view(-1)

        dn_label_noise_ratio = self.dn_label_noise_ratio
        knwon_labels_expand = known_labels.clone()
        if dn_label_noise_ratio > 0:
            prob = torch.rand_like(knwon_labels_expand.float())
            chosen_indice = prob < dn_label_noise_ratio
            new_label = torch.randint_like(knwon_labels_expand[chosen_indice], 0,
                                           self.num_classes)  # randomly put a new one here
            # gt_labels_expand.scatter_(0, chosen_indice, new_label)
            knwon_labels_expand[chosen_indice] = new_label

        known_labels = self.label_enc(knwon_labels_expand)
        # noised_known_features = known_labels

        # known_features = torch.cat([(m * mask_features[i]).flatten(-2).mean(-1) for i, m in enumerate(masks_)])
        # known_features = known_features.repeat(scalar, 1).detach()

        noised_known_features = known_labels


        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')

        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_boxes])  # [1,2, 1,2,3]
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(scalar)]).long().cuda()

        # for i in range(bs):
        padding[(known_bid, map_known_indices)] = noised_known_features

        for i, padding_mask in enumerate(padding_mask_3level):
            padding_mask_3level[i][(known_bid, map_known_indices)] = masks_3level[2-i]
            padding_mask_3level[i] = padding_mask_3level[i].unsqueeze(1).repeat([1, 8, 1, 1])

        res = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        res = torch.cat([padding.transpose(0, 1), res], dim=0)
        ###################

        outputs_class, outputs_mask, attn_mask_ = self.forward_prediction_heads(res, mask_features,
                                                                                attn_mask_target_size=size_list[0])
        attn_mask_ = attn_mask_.reshape([bs, 8, -1, attn_mask_.shape[-1]])
        attn_mask_[:, :, :-self.num_queries] = padding_mask_3level[0]
        attn_mask_ = attn_mask_.flatten(0, 1)
        #####################

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return None, res, attn_mask, dn_args_, outputs_class, outputs_mask, attn_mask_, padding_mask_3level
        # pass

    def prepare_for_dn_v8(self, mask_features, dn_args, size_list):
        targets, scalar, noise_scale = dn_args['tgt'], dn_args['scalar'], dn_args['noise_scale']
        boxes = [targets[i]['boxes'] for i in range(len(targets))]
        num_boxes = [len(b) for b in boxes]
        single_pad = max_num = max(num_boxes)
        if max_num == 0 or scalar == 0:
            return None
        if scalar >= 100:
            scalar = scalar // max_num
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size
        bs = len(boxes)

        padding = torch.zeros([bs, pad_size, self.query_feat.weight.shape[-1]]).cuda()
        padding_mask_3level = []
        for i in range(len(size_list)):
            padding_mask = torch.ones([bs, pad_size, size_list[i][0] * size_list[i][1]]).cuda().bool()
            padding_mask_3level.append(padding_mask)

        boxes = torch.cat([t['boxes'] for t in targets])  # x, y, x, y
        knwon_boxes_expand = boxes.repeat(scalar, 1)

        # knwon_labels_expand = known_labels.clone()
        dn_mask_noise_scale_shift = noise_scale
        dn_mask_noise_scale_scale = noise_scale
        masks_3level = []
        if dn_mask_noise_scale_shift > 0:
            masks = torch.cat(
                [F.interpolate(targets[i]['masks'].float().unsqueeze(1), size=size_list[-1], mode="nearest")
                 for i in range(len(targets))]).flatten(0, 1)
            masks = masks.repeat(scalar, 1, 1)
            edge_order=torch.rand_like(knwon_boxes_expand[...,0])
            new_masks = []
            masks = masks < 0.5
            areas = torch.clamp((~masks).sum(dim=[1,2]).float()*noise_scale,min=1.0)
            for mask,eo,area in zip(masks,edge_order,areas):
                max_short_edges=torch.sqrt(area).long()
                short_=random.randint(1,max_short_edges.item())
                long_=int(area/short_)
                if eo < 0.5:
                    patch_h=short_
                    patch_w=min(long_,mask.shape[1])
                else:
                    patch_h = min(long_,mask.shape[0])
                    patch_w = short_
                x0=random.randint(0,mask.shape[0]-patch_h)
                y0=random.randint(0,mask.shape[1]-patch_w)
                mask[x0:x0+patch_h,y0:y0+patch_w]=False

                new_masks.append(mask)
            new_masks = torch.stack(new_masks)
            masks = new_masks.flatten(1)

            masks_3level.append(masks)
            # import ipdb; ipdb.set_trace()
            masks = (F.interpolate(new_masks.unsqueeze(1).float(), size=size_list[-2], mode="nearest") > 0.5).flatten(1)
            masks_3level.append(masks)

            masks = (F.interpolate(new_masks.unsqueeze(1).float(), size=size_list[0], mode="nearest") > 0.5).flatten(1)
            masks_3level.append(masks)

            # knwon_boxes_expand = knwon_boxes_expand.clamp(min=0.0, max=1.0)

        labels = torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(scalar, 1).view(-1)

        dn_label_noise_ratio = self.dn_label_noise_ratio
        knwon_labels_expand = known_labels.clone()
        if dn_label_noise_ratio > 0:
            prob = torch.rand_like(knwon_labels_expand.float())
            chosen_indice = prob < dn_label_noise_ratio
            new_label = torch.randint_like(knwon_labels_expand[chosen_indice], 0,
                                           self.num_classes)  # randomly put a new one here
            # gt_labels_expand.scatter_(0, chosen_indice, new_label)
            knwon_labels_expand[chosen_indice] = new_label

        known_labels = self.label_enc(knwon_labels_expand)
        # noised_known_features = known_labels

        # known_features = torch.cat([(m * mask_features[i]).flatten(-2).mean(-1) for i, m in enumerate(masks_)])
        # known_features = known_features.repeat(scalar, 1).detach()
        # if dn_label_noise_ratio > 0:
        noised_known_features = known_labels
        # else:
        #     dn_delta = (torch.rand_like(known_labels) * 2 - 1.0) * noise_scale * known_labels
        #     noised_known_features = known_labels + dn_delta

        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')

        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_boxes])  # [1,2, 1,2,3]
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(scalar)]).long().cuda()

        # for i in range(bs):
        padding[(known_bid, map_known_indices)] = noised_known_features

        for i, padding_mask in enumerate(padding_mask_3level):
            padding_mask_3level[i][(known_bid, map_known_indices)] = masks_3level[2-i]
            padding_mask_3level[i] = padding_mask_3level[i].unsqueeze(1).repeat([1, 8, 1, 1])

        res = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        res = torch.cat([padding.transpose(0, 1), res], dim=0)
        ###################

        outputs_class, outputs_mask, attn_mask_ = self.forward_prediction_heads(res, mask_features,
                                                                                attn_mask_target_size=size_list[0])
        attn_mask_ = attn_mask_.reshape([bs, 8, -1, attn_mask_.shape[-1]])
        attn_mask_[:, :, :-self.num_queries] = padding_mask_3level[0]
        attn_mask_ = attn_mask_.flatten(0, 1)
        #####################

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return None, res, attn_mask, dn_args_, outputs_class, outputs_mask, attn_mask_, padding_mask_3level
        # pass

    def prepare_for_dn_v9(self, mask_features, dn_args, size_list):
        targets, scalar, noise_scale = dn_args['tgt'], dn_args['scalar'], dn_args['noise_scale']
        boxes = [targets[i]['boxes'] for i in range(len(targets))]
        num_boxes = [len(b) for b in boxes]
        single_pad = max_num = max(num_boxes)
        if max_num == 0 or scalar == 0:
            return None
        if scalar >= 100:
            scalar = scalar // max_num
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size
        bs = len(boxes)

        padding = torch.zeros([bs, pad_size, self.query_feat.weight.shape[-1]]).cuda()
        padding_mask_3level = []
        for i in range(len(size_list)):
            padding_mask = torch.ones([bs, pad_size, size_list[i][0] * size_list[i][1]]).cuda().bool()
            padding_mask_3level.append(padding_mask)

        boxes = torch.cat([t['boxes'] for t in targets])  # x, y, x, y
        knwon_boxes_expand = boxes.repeat(scalar, 1)

        # knwon_labels_expand = known_labels.clone()


        masks_3level = []
        if noise_scale > 0:
            masks = torch.cat(
                [F.interpolate(targets[i]['masks'].float().unsqueeze(1), size=size_list[-1], mode="nearest")
                 for i in range(len(targets))]).flatten(0, 1)
            masks = masks.repeat(scalar, 1, 1)
            edge_order = torch.rand_like(knwon_boxes_expand[..., 0])
            new_masks = []
            masks = masks < 0.5
            noise= torch.rand_like(masks.float())
            noise=noise<noise_scale
            new_masks=torch.logical_or(masks,noise)

            masks_3level.append(new_masks.flatten(1))
            # import ipdb; ipdb.set_trace()
            masks = (F.interpolate(masks.unsqueeze(1).float(), size=size_list[-2],
                                   mode="nearest") > 0.5).flatten(0, 1)

            noise = torch.rand_like(masks.float())
            noise = noise < noise_scale
            new_masks = torch.logical_or(masks, noise)

            masks_3level.append(new_masks.flatten(1))

            masks = (F.interpolate(masks.unsqueeze(1).float(), size=size_list[0],
                                   mode="nearest") > 0.5).flatten(0, 1)
            noise = torch.rand_like(masks.float())
            noise = noise < noise_scale
            new_masks = torch.logical_or(masks, noise)
            masks_3level.append(new_masks.flatten(1))

            # knwon_boxes_expand = knwon_boxes_expand.clamp(min=0.0, max=1.0)

        labels = torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(scalar, 1).view(-1)

        dn_label_noise_ratio = self.dn_label_noise_ratio
        knwon_labels_expand = known_labels.clone()
        if dn_label_noise_ratio > 0:
            prob = torch.rand_like(knwon_labels_expand.float())
            chosen_indice = prob < dn_label_noise_ratio
            new_label = torch.randint_like(knwon_labels_expand[chosen_indice], 0,
                                           self.num_classes)  # randomly put a new one here
            # gt_labels_expand.scatter_(0, chosen_indice, new_label)
            knwon_labels_expand[chosen_indice] = new_label

        known_labels = self.label_enc(knwon_labels_expand)
        # noised_known_features = known_labels

        # known_features = torch.cat([(m * mask_features[i]).flatten(-2).mean(-1) for i, m in enumerate(masks_)])
        # known_features = known_features.repeat(scalar, 1).detach()
        # if dn_label_noise_ratio > 0:
        noised_known_features = known_labels
        # else:
        #     dn_delta = (torch.rand_like(known_labels) * 2 - 1.0) * noise_scale * known_labels
        #     noised_known_features = known_labels + dn_delta

        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')

        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_boxes])  # [1,2, 1,2,3]
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(scalar)]).long().cuda()

        # for i in range(bs):
        padding[(known_bid, map_known_indices)] = noised_known_features

        for i, padding_mask in enumerate(padding_mask_3level):
            padding_mask_3level[i][(known_bid, map_known_indices)] = masks_3level[2 - i]
            padding_mask_3level[i] = padding_mask_3level[i].unsqueeze(1).repeat([1, 8, 1, 1])

        res = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        res = torch.cat([padding.transpose(0, 1), res], dim=0)
        ###################

        outputs_class, outputs_mask, attn_mask_ = self.forward_prediction_heads(res, mask_features,
                                                                                attn_mask_target_size=size_list[0])
        attn_mask_ = attn_mask_.reshape([bs, 8, -1, attn_mask_.shape[-1]])
        attn_mask_[:, :, :-self.num_queries] = padding_mask_3level[0]
        attn_mask_ = attn_mask_.flatten(0, 1)
        #####################

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return None, res, attn_mask, dn_args_, outputs_class, outputs_mask, attn_mask_, padding_mask_3level
        # pass
        # pass

    def gen_mask_dn(self,dn_args,size_list,known_bid,map_known_indices):
        targets, scalar, noise_scale = dn_args['tgt'], dn_args['scalar'], dn_args['noise_scale']
        boxes = [targets[i]['boxes'] for i in range(len(targets))]
        num_boxes = [len(b) for b in boxes]
        max_num = max(num_boxes)
        if max_num == 0 or scalar == 0:
            return None
        if scalar >= 100:
            scalar = scalar // max_num
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size
        bs = len(boxes)

        padding_mask = torch.ones([bs, pad_size, size_list[0] * size_list[1]]).cuda().bool()
        masks = torch.cat(
            [F.interpolate(targets[i]['masks'].float().unsqueeze(1), size=size_list, mode="area").flatten(1) <= 1e-8
             for i in range(len(targets)) if len(targets[i]['masks']) > 0]).repeat(scalar, 1)
        if self.head_dn:
            masks = masks[:, None].repeat(1, 8, 1)
            areas = (~masks).sum(2)
            noise_ratio = areas * noise_scale / (size_list[0] * size_list[1])
            delta_mask = torch.rand_like(masks, dtype=torch.float) < noise_ratio[:, :, None]
            masks = torch.logical_xor(masks, delta_mask)  # [q,h,h*w]
        else:
            areas = (~masks).sum(1)
            noise_ratio = areas * noise_scale / (size_list[0] * size_list[1])
            delta_mask = torch.rand_like(masks, dtype=torch.float) < noise_ratio[:, None]
            masks = torch.logical_xor(masks, delta_mask)

        if self.head_dn:
            padding_mask = padding_mask.unsqueeze(2).repeat([1, 1, 8, 1])  # [bs,q,h,h*w]
            padding_mask[(known_bid, map_known_indices)] = masks
            padding_mask = padding_mask.transpose(1, 2)
        else:
            padding_mask[(known_bid, map_known_indices)] = masks
            padding_mask = padding_mask.unsqueeze(1).repeat([1, 8, 1, 1])
        return padding_mask

    def prepare_for_dn(self, mask_features, dn_args,size_list):
        targets, scalar, noise_scale = dn_args['tgt'], dn_args['scalar'], dn_args['noise_scale']
        boxes = [targets[i]['boxes'] for i in range(len(targets))]
        num_boxes = [len(b) for b in boxes]
        single_pad = max_num = max(num_boxes)
        scalar=1
        if max_num == 0 or scalar==0:
            return None
        if scalar >= 100:
            scalar = scalar // max_num
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size
        bs = len(boxes)

        padding = torch.zeros([bs, pad_size, self.query_feat.weight.shape[-1]]).cuda()
        padding_mask = torch.ones([bs, pad_size, size_list[0][0]*size_list[0][1]]).cuda().bool()
        # padding[]
        # gt_masks=torch.cat([])
        # import pdb;pdb.set_trace()
        #attn_mask[B * h, Q, HW]
        #masks [b*h,pad_size,hw]
        # masks_ = [F.interpolate(targets[i]['masks'].unsqueeze(1).float(), size=mask_features.shape[-2:], mode="bilinear")
        #          for i in range(len(targets))]
        masks = torch.cat([F.interpolate(targets[i]['masks'].unsqueeze(1), size=size_list[0], mode="nearest").flatten(1)<0.5
                 for i in range(len(targets))]).repeat(scalar, 1)

        labels=torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(scalar, 1).view(-1)
        known_labels = self.label_enc(known_labels)
        noised_known_features = known_labels

        # known_features = torch.cat([(m * mask_features[i]).flatten(-2).mean(-1) for i, m in enumerate(masks_)])
        # known_features = known_features.repeat(scalar, 1).detach()

        # dn_delta = (torch.rand_like(known_features) * 2 - 1.0) * noise_scale * known_features
        # noised_known_features = known_features + dn_delta

        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')

        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_boxes])  # [1,2, 1,2,3]
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(scalar)]).long().cuda()

        # for i in range(bs):
        padding[(known_bid, map_known_indices)] = noised_known_features
        padding_mask[(known_bid, map_known_indices)]=masks
        padding_mask=padding_mask.unsqueeze(1).repeat([1,8,1,1])

        res = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        res = torch.cat([padding.transpose(0, 1), res], dim=0)
        ###################

        outputs_class, outputs_mask, attn_mask_ = self.forward_prediction_heads(res, mask_features,attn_mask_target_size=size_list[0])
        attn_mask_=attn_mask_.view([bs,8,-1,attn_mask_.shape[-1]])
        attn_mask_[:,:,:-self.num_queries]=padding_mask
        attn_mask_=attn_mask_.flatten(0,1)
        #####################

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return None, res, attn_mask, dn_args_,outputs_class, outputs_mask, attn_mask_
        # pass

    def postprocess_for_dn(self, predictions_class, predictions_mask):
        n_lys = len(predictions_class)
        dn_predictions_class, predictions_class = [predictions_class[i][:, :-self.num_queries] for i in range(n_lys)], \
                                                  [predictions_class[i][:, -self.num_queries:] for i in range(n_lys)]
        dn_predictions_mask, predictions_mask = [predictions_mask[i][:, :-self.num_queries] for i in range(n_lys)], \
                                                [predictions_mask[i][:, -self.num_queries:] for i in range(n_lys)]
        return predictions_class, predictions_mask, dn_predictions_class, dn_predictions_mask
        # pass

    def forward(self, x, mask_features, mask=None, dn_args=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        if dn_args is not None:
            if self.dn_mode=='base':
                res= self.prepare_for_dn(mask_features, dn_args,size_list)
            elif self.dn_mode=='lb':
                res= self.prepare_for_dn_v2(mask_features, dn_args,size_list)
            elif self.dn_mode=='mask':
                res=self.prepare_for_dn_v3(mask_features, dn_args,size_list)
            elif self.dn_mode=='points':
                res= self.prepare_for_dn_v5(mask_features, dn_args,size_list)
            # elif self.dn_mode=='multi_ly_lb':
            #     res= self.prepare_for_dn_v5(mask_features, dn_args,size_list)
            elif self.dn_mode=='shift':
                res= self.prepare_for_dn_v6(mask_features, dn_args,size_list)
            elif self.dn_mode=='scale':
                res= self.prepare_for_dn_v7(mask_features, dn_args,size_list)
            elif self.dn_mode=='shift_scale':
                res= self.prepare_for_dn_v7(mask_features, dn_args,size_list,shift_dn=True)
            elif self.dn_mode=='patch':
                res= self.prepare_for_dn_v8(mask_features, dn_args,size_list)
            elif self.dn_mode=='points_MAE':
                res= self.prepare_for_dn_v9(mask_features, dn_args,size_list)
            else:
                res= None
            if res is None:
                query_embed, output, tgt_mask, outputs_class, outputs_mask, attn_mask = \
                    self.prepare_for_normal(bs, mask_features, size_list)
            else:
                if self.dn_mode!='points':
                    query_embed, output, tgt_mask, dn_args_, outputs_class, outputs_mask, attn_mask,padding_mask_3level\
                        = res
                else:
                    query_embed, output, tgt_mask, dn_args_,outputs_class, outputs_mask, attn_mask=res
                    known_bid,map_known_indices=query_embed
                    query_embed=None
        else:
            query_embed, output, tgt_mask, outputs_class, outputs_mask, attn_mask=\
                self.prepare_for_normal(bs,mask_features,size_list)
        # import pdb;pdb.set_trace()

        predictions_class = []
        predictions_mask = []
        dn_predictions_class = []
        dn_predictions_mask = []

        # prediction heads on learnable query features
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        last_q=None
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # import pdb;pdb.set_trace()
            # Deal with extreme cases when all elements are masked
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # import pdb;
            # pdb.set_trace()
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=tgt_mask,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )
            # attn_mask:[B*h, Q, HW]

            if self.all_lys:
                flag=True
            else:
                flag=i<3
            level = (i + 1) % self.num_feature_levels

            if dn_args is not None and tgt_mask is not None and flag:
                if self.dn_mode=="points":
                    outputs_class, outputs_mask, attn_mask,last_q = self.forward_prediction_heads(output, mask_features,
                                                                                           attn_mask_target_size=size_list[level],last_q=last_q)
                    padding_mask=self.gen_mask_dn(dn_args,size_list=size_list[(i + 1) % self.num_feature_levels],known_bid=known_bid,map_known_indices=map_known_indices)
                    attn_mask = attn_mask.view([bs, 8, -1, attn_mask.shape[-1]])
                    attn_mask[:, :, :-self.num_queries] = padding_mask
                    attn_mask = attn_mask.flatten(0, 1)
                else:
                    outputs_class, outputs_mask, attn_mask,last_q = self.forward_prediction_heads_dn(output, mask_features,
                                                                                           attn_mask_target_size=size_list[level]
                                                                                           ,padding_mask=padding_mask_3level[level],last_q=last_q)
            else:
                outputs_class, outputs_mask, attn_mask,last_q= self.forward_prediction_heads(output, mask_features,
                                                                                       attn_mask_target_size=size_list[
                                                                                           level],last_q=last_q
                                                                                       )

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1
        if not (tgt_mask is None):
            predictions_class, predictions_mask, dn_predictions_class, dn_predictions_mask = \
                self.postprocess_for_dn(predictions_class, predictions_mask)

            dn_out = {
                'pred_logits': dn_predictions_class[-1],
                'pred_masks': dn_predictions_mask[-1],
                'aux_outputs': self._set_aux_loss(
                    dn_predictions_class if self.mask_classification else None, dn_predictions_mask
                ),
                'dn_args': dn_args_

            }
        else:
            dn_out = None
            predictions_class[-1]+=self.label_enc.weight[0,0]*0.0

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
            'dn_out': dn_out
        }

        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size,last_q=None):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        # import pdb;pdb.set_trace()
        if not last_q is None:
            mask_embed=self.last_q_ratio*last_q+(1-self.last_q_ratio)*mask_embed
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # import pdb;pdb.set_trace()
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                         1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask,mask_embed.detach()

    def forward_prediction_heads_dn(self, output, mask_features, attn_mask_target_size,padding_mask=None,last_q=None):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        # import pdb;pdb.set_trace()
        if not last_q is None:
            mask_embed=self.last_q_ratio*last_q+(1-self.last_q_ratio)*mask_embed
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # import pdb;pdb.set_trace()
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        if padding_mask is not None:
            attn_mask = attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn_mask[:,:,:-self.num_queries] = padding_mask
            attn_mask= (attn_mask.flatten(0,1) < 0.5).bool()
        else:
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()


        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask,mask_embed.detach()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]