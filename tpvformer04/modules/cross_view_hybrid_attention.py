
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
import math
from mmcv.runner.base_module import BaseModule

from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class TPVCrossViewHybridAttention(BaseModule):
    """Cross view hybrid attention module used in TPVFormer.
    Based on deformable attention.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 num_tpv_queue=2):

        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        # num_tpv_queue: 历史查询个数，简单认为是 batchsize 的一部分吧
        self.num_tpv_queue = num_tpv_queue
        # 采样偏移权重矩阵: MLK2
        self.sampling_offsets = nn.Linear(
            embed_dims * num_tpv_queue, num_tpv_queue * num_heads * num_levels * num_points * 2)
        # 注意力分数权重矩阵: MLK1 查询用此矩阵投影再 softmax 得到标量注意力分数
        self.attention_weights = nn.Linear(embed_dims * num_tpv_queue,
                                           num_tpv_queue * num_heads * num_levels * num_points)
        # 值权重矩阵: 用于将查询向量投影到值空间
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        # 多头权重矩阵: 用于将每个注意力头的输出投影到最终输出空间
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels*self.num_tpv_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query: Size(B, num_query=H*W, embed_dims)
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [bs, num_query, embed_dims].
        """

        if value is None:
            # 这里重复两遍的 2 应等于 num_tpv_queue
            # Sizee(2B, num_query=H*W=10000, embed_dims=256)
            value = torch.cat([query, query], 0)

        if identity is None:
            identity = query
        # @# 位置嵌入，俯视栅格查询与其展平的二维位置嵌入向量做加法
        if query_pos is not None:
            # Size(B, num_query, embed_dims)
            query = query + query_pos
        if not self.batch_first:
            # 不执行，已经是 B 在前
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        bs,  num_query, _ = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_tpv_queue == 2

        # -> Size(B, num_query, (B+1)*embed_dims)
        query = torch.cat([value[:bs], query], -1)
        # @# 查询向量投影到值空间，这里的值就是原查询向量
        value = self.value_proj(value)
        # Size(2B, num_query, embed_dims) -> Size(2B, num_query, num_heads=8, C=32) 
        value = value.reshape(self.num_tpv_queue*bs,
                              num_value, self.num_heads, -1)

        # @# 查询向量投影得到采样偏移 Size(B, num_query, (B+1)*embed_dims) -> Size(B, num_query, C=128)
        sampling_offsets = self.sampling_offsets(query)
        # -> Size(B, num_query, num_heads=8, num_tpv_queue=2, num_levels=1, num_points=4, 2)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_tpv_queue, self.num_levels, self.num_points, 2)
        # @# 查询向量投影再 softmax 得到标量注意力分数，attention_weights 为 Size(B, num_query, num_heads=8, num_tpv_queue=2, num_points=4)
        attention_weights = self.attention_weights(query).view(
            bs, num_query,  self.num_heads, self.num_tpv_queue, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_tpv_queue,
                                                   self.num_levels,
                                                   self.num_points)

        attention_weights = attention_weights.permute(3, 0, 1, 2, 4, 5)\
            .reshape(bs*self.num_tpv_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(3, 0, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_tpv_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        # @# 参考点加采样偏移并归一化坐标
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            # reference_points: Size(2B, num_query, 1, 2) -> sampling_locations: Size(2B, num_query, num_heads=8, 1, num_points=4, 2)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        # @# 执行多尺度可变形注意力
        if torch.cuda.is_available() and value.is_cuda:

            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        # output shape (bs*num_tpv_queue, num_query, embed_dims)
        # (bs*num_tpv_queue, num_query, embed_dims)-> (num_query, embed_dims, bs*num_tpv_queue)
        output = output.permute(1, 2, 0)

        # fuse history value and current value
        # (num_query, embed_dims, bs*num_tpv_queue)-> (num_query, embed_dims, bs)
        output = (output[..., :bs] + output[..., bs:])/self.num_tpv_queue

        # (num_query, embed_dims, bs)-> (bs, num_query, embed_dims)
        output = output.permute(2, 0, 1)

        # 将每个注意力头的输出投影到最终输出空间 (bs, num_query=10000, embed_dims=256)
        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
