代码笔记文件语法规范：**强调** 语法用来标识重点关注的代码，~~删除线~~ 语法用来标识可忽略的代码，<datastruct> HTML 语法表示引用的数据结构，[module] 链接语法用来标识链接到模块代码，<!-- cond --> 注释语法用来标识代码执行条件

#### MMDet2.14.0 模块实现层
- LearnedPositionalEncoding: 二维位置嵌入，nn.Embedding 是一维的
  - \__init__: 用 nn.Embedding 构造行嵌入和列嵌入
    """
    Args:
      num_feats: x 或 y 单个轴的嵌入特征维度
      row_num_embed/col_num_embed: 行/列嵌入个数
    """
  - forward: 输入掩膜作为索引，输出二维位置嵌入向量
    """
    用 arang 生成从零递增的整数索引 y/x 并从行/列嵌入索引行/列嵌入向量
    行/列嵌入向量分别广播为 (H, W, C=num_feats) 通道拼接为 (H, W, C=2*num_feats) 变形为 (B, C=2*num_feats, H, W) 输出
    Args:
      mask: Size(B, H, W) 二进制掩膜，值为 0 处为有效位置才索引其嵌入向量，这点与直觉相反
    Returns:
      pos: Size(B, C=2*num_feats, H, W)
    """

#### 代码框架图
<!-- 常规 Pytorch 训练框架，略 -->
- python train.py --py-config config/tpv04_occupancy.py --work-dir out/tpv_occupancy
  - data_builder.build/([ImagePoint_NuScenes] + [DatasetWrapper_NuScenes])
  - [tpvformer04/TPVFormer]
- python train.py --py-config config/tpv_lidarseg.py --work-dir out/tpv_lidarseg
- python eval.py --py-config config/tpv_lidarseg.py --ckpt-path model_zoo/download/tpv10_lidarseg_v2.pth
#### 模块实现层
<!-- 数据集和数据变换 -->
- ImagePoint_NuScenes: 相当于数据集基类，但是却不是以继承的方式，而是将实例作为形参传到其他数据集实例中
  - \__init__: 读 pkl 文件加载数据信息列表、读 yaml 文件加载标签映射数组
  - \__getitem__: 获取数据信息、读图片读点云读点云标签、标签映射（合并原始标签的某些类别到只有 16 类）
    - get_data_info: 额外计算激光雷达到像素坐标系的变换矩阵等
- DatasetWrapper_NuScenes
  - \__init__/[TRANSFORMS]: 构造数据变换
  - __getitem__: <ds_occ_data_tuple>
    从子数据集获取原始数据（图像点云等）
    执行数据变换，都是对图像的数据增强
    计算点云点的体素坐标 grid_ind
    计算体素标签 processed_label
    - ImagePoint_NuScenes.\__getitem__
    - [TRANSFORMS].\__call__
    - **nb_process_label**: 当前搜索体素 i，计数器计数 i 内的点标签直方图，遍历每个点云点，点云点体素 j，如果 ij 坐标相等则计数器加 1，不相等则表明 i 内点已被搜索完，则取点标签众数作为体素标签，计数器清零并令 i=j 搜索下一个体素

- **PhotoMetricDistortionMultiViewImage**: 逐个图像执行亮度随机扰动、对比度随机扰动等一系列数据增强，参考笔记 [CN/00002]
- NormalizeMultiviewImage: 逐个图像归一化，即像素值减均值除标准差，这里像素值值域为 0 到 255，减完均值后像素值分布以 0 为中心，像素值还是能正负大几十
- RandomScaleImageMultiViewImage:
- PadMultiViewImage: 用 0 填充至图像宽高均为 32 的倍数
<!-- 模型 -->
- tpvformer04/TPVFormer/BaseModule
  - \__init__
    - TPVFormerHead: 构建三个视图的栅格查询（一维嵌入）等
      - [mmdet.models.utils.positional_encoding.LearnedPositionalEncoding]
      - TPVFormerEncoder: 计算三个视图上的三维参考点和二维参考点
        """
        Locals:
          ref_3d_hw/ref_3d_zh/ref_3d_wz: 俯视/侧视/正视平面的三维参考点，即柱子上均匀取的几个点
          ref_2d_hw/ref_2d_zh/ref_2d_wz: 俯视/侧视/正视平面的二维参考点，即平面二维栅格的中心点集
        """
        - mmcv.cnn.bricks.transformer.TransformerLayerSequence/[TPVFormerLayer]
        - get_reference_points: 获取参考点，① 三维参考点，相当于对单位立方体，按 (num_points_in_pillar, H, W) 的尺寸体素化，每个体素中心点即为采样点，输出 Size(B, num_points_in_pillar, H*W, 3) ② 二维参考点同理 (H,W) 二维栅格的中心点，输出 Size(B, H*W, 1, 2)
    - ResNet + FPN + GridMask
  - forward
    - extract_img_feat: 网格掩码数据变换、ResNet101 提取多尺度特征、FPN 多尺度特征融合
      - GridMask.forward: 相当于根据条幅宽度 d 用白刷子横竖（先按"三"字画再按"川"字画）将图像刷白，再根据掩盖宽度 l 用黑刷子（黑刷子"笔画细"(l小于d)且在白刷子笔画中间(有偏移量)）横竖将图像刷黑，最后黑白取反可得到格子状的掩膜
      - ResNet.forward: 返回后三个阶段的多个尺度的特征图
      - FPN.forward: 返回四个尺度的特征图
    - **TPVFormerHead.forward**
      """
      广播栅格查询（一维嵌入）的参数矩阵 + 获取二维位置嵌入向量
      添加相机号和特征图层级号嵌入，每个二维特征图展平为一维向量（即合并 HW 通道）并和广播后的相机号嵌入、特征图层级号嵌入做加法
      多个尺寸的特征图进一步展平为一个一维向量后送入编码器 [Size(6, B, H*W=23200, C=256),...,len(4)] -> Size(6, B, HW++=30825, C=256)
      """
      - LearnedPositionalEncoding: 输入掩膜作为索引，输出二维位置嵌入向量，掩膜随训练更新，这里输出 Size(B, C=256, H=100, W=100)
      - TPVFormerEncoder.forward: 遍历三个视角的三维参考点分别获取其对应的参考像素坐标、顺序执行 TPVFormerLayer 层（这里为相同配置的三层）
        - point_sampling: 返回该视角三维参考点对应的参考像素坐标，Size(num_cam, B, num_query, D=num_points_in_pillar, 2)
        - [TPVFormerLayer].forward

- TPVFormerLayer: 预备知识 2010.DeformableDETR
  - \__init__
    - build_attention
      - TPVCrossViewHybridAttention
      - TPVImageCrossAttention/TPVMSDeformableAttention3D
    - build_feedforward_network/FFN
    - build_norm_layer/LN
  - forward
    - **TPVCrossViewHybridAttention.forward**: 多尺度可变形注意力
      """
      位置嵌入，俯视栅格查询与其展平的二维位置嵌入向量做加法
      查询向量投影到值空间
      查询向量投影得到采样偏移
      查询向量投影再 softmax 得到标量注意力分数
      参考点加采样偏移并归一化坐标
      Args:
        query: Size(B, num_query=H*W, embed_dims) 来自于 TPVFormerHead 的俯视图栅格查询
      """
      - MultiScaleDeformableAttnFunction_fp32: 执行多尺度可变形注意力
    - LN
    - TPVImageCrossAttention.forward
      """
      
      """
      - TPVMSDeformableAttention3D.forward
    - LN + FFN + LN