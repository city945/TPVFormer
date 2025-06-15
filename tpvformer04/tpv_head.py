
import torch
import torch.nn as nn
from torch.nn.init import normal_

from mmcv.runner import BaseModule
from mmseg.models import HEADS
from mmcv.cnn.bricks.transformer import build_positional_encoding, \
    build_transformer_layer_sequence
from mmcv.runner import force_fp32, auto_fp16

from .modules.cross_view_hybrid_attention import TPVCrossViewHybridAttention
from .modules.image_cross_attention import TPVMSDeformableAttention3D


@HEADS.register_module()
class TPVFormerHead(BaseModule):

    def __init__(self,
                 positional_encoding=None,
                 tpv_h=30,
                 tpv_w=30,
                 tpv_z=30,
                 pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
                 num_feature_levels=4,
                 num_cams=6,
                 encoder=None,
                 embed_dims=256,
                 **kwargs):
        super().__init__()

        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.real_z = self.pc_range[5] - self.pc_range[2]
        self.fp16_enabled = False

        # positional encoding
        self.positional_encoding = build_positional_encoding(positional_encoding)
        # tpv_mask_hw: 掩膜，用于 LearnedPositionalEncoding，全零表示所有位置都有效
        tpv_mask_hw = torch.zeros(1, tpv_h, tpv_w)
        self.register_buffer('tpv_mask_hw', tpv_mask_hw)

        # transformer layers
        self.encoder = build_transformer_layer_sequence(encoder)
        # @! 位置嵌入：即查找表，输入索引输出嵌入向量。有两种实现方式 nn.Parameter 和 nn.Embedding，前者直接定义参数矩阵(尺寸为 (N=嵌入总数, D=嵌入维度))，后者类内部定义参数矩阵并维护一个查找表，区别在于前者需要手动输入单个索引后者可以输入索引张量、前者可以手动操作参数矩阵如广播等
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        # 构建三个视图的栅格查询（即一维嵌入），栅格尺寸 100*100*8
        self.tpv_embedding_hw = nn.Embedding(self.tpv_h * self.tpv_w, self.embed_dims)
        self.tpv_embedding_zh = nn.Embedding(self.tpv_z * self.tpv_h, self.embed_dims)
        self.tpv_embedding_wz = nn.Embedding(self.tpv_w * self.tpv_z, self.embed_dims)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, TPVMSDeformableAttention3D) or isinstance(m, TPVCrossViewHybridAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        """
        bs = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device

        # @# 广播栅格查询（一维嵌入）的参数矩阵
        # tpv queries and pos embeds
        tpv_queries_hw = self.tpv_embedding_hw.weight.to(dtype)
        tpv_queries_zh = self.tpv_embedding_zh.weight.to(dtype)
        tpv_queries_wz = self.tpv_embedding_wz.weight.to(dtype)
        # (N=H*W=1000, C=256) -> (B, N=1000, C=256)
        tpv_queries_hw = tpv_queries_hw.unsqueeze(0).repeat(bs, 1, 1)
        tpv_queries_zh = tpv_queries_zh.unsqueeze(0).repeat(bs, 1, 1)
        tpv_queries_wz = tpv_queries_wz.unsqueeze(0).repeat(bs, 1, 1)
        # @# 获取二维位置嵌入向量
        tpv_mask_hw = self.tpv_mask_hw.expand(bs, -1, -1)
        # 输入掩膜作为索引，输出二维位置嵌入向量，掩膜随训练更新，这里输出 Size(B, C=256, H=100, W=100)
        tpv_pos_hw = self.positional_encoding(tpv_mask_hw).to(dtype)
        # -> Size(B, N=H*W=10000, C=256)
        tpv_pos_hw = tpv_pos_hw.flatten(2).transpose(1, 2)
        
        # @# 添加相机号和特征图层级号嵌入，每个二维特征图展平为一维向量（即合并 HW 通道）并和广播后的相机号嵌入、特征图层级号嵌入做加法
        # [Size(B, 6, C=256, H=116, W=200),...] -> [Size(6, B, H*W=23200, C=256),...]
        # flatten image features of different scales
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2) # num_cam, bs, hw, c
            feat = feat + self.cams_embeds[:, None, None, :].to(dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl+1, :].to(dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        # @# 多个尺寸的特征图进一步展平为一个一维向量后送入编码器 [Size(6, B, H*W=23200, C=256),...,len(4)] -> Size(6, B, HW++=30825, C=256)
        feat_flatten = torch.cat(feat_flatten, 2) # num_cam, bs, hw++, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        tpv_embed = self.encoder(
            [tpv_queries_hw, tpv_queries_zh, tpv_queries_wz],
            feat_flatten,
            feat_flatten,
            tpv_h=self.tpv_h,
            tpv_w=self.tpv_w,
            tpv_z=self.tpv_z,
            tpv_pos=[tpv_pos_hw, None, None],
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            img_metas=img_metas,
        )
        
        return tpv_embed