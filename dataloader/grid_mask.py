import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from mmcv.runner import force_fp32, auto_fp16

class Grid(object):
    def __init__(self, use_h, use_w, rotate = 1, offset=False, ratio = 0.5, mode=0, prob = 1.):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode=mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, img, label):
        if np.random.rand() > self.prob:
            return img, label
        h = img.size(1)
        w = img.size(2)
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5*h)
        ww = int(1.5*w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d*self.ratio+0.5),1),d-1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh//d):
                s = d*i + st_h
                t = min(s+self.l, hh)
                mask[s:t,:] *= 0
        if self.use_w:
            for i in range(ww//d):
                s = d*i + st_w
                t = min(s+self.l, ww)
                mask[:,s:t] *= 0
       
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]

        mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1-mask

        mask = mask.expand_as(img)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h,w) - 0.5)).float()
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask 

        return img, label


class GridMask(nn.Module):
    def __init__(self, use_h, use_w, rotate = 1, offset=False, ratio = 0.5, mode=0, prob = 1.):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.fp16_enable = False
    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch #+ 1.#0.5
    @auto_fp16()
    def forward(self, x):
        """
        相当于根据条幅宽度 d 用白刷子横竖（先按"三"字画再按"川"字画）将图像刷白，再根据掩盖宽度 l 用黑刷子（黑刷子"笔画细"(l小于d)且在白刷子笔画中间(有偏移量)）横竖将图像刷黑，最后黑白取反可得到格子状的掩膜
        """
        # @Todo 应该作为数据变换而不是在 GPU 上执行
        # 可视化输入图像: pu4c.det3d.app.image_viewer(data=np.transpose(x[3].cpu().numpy(), (1, 2, 0)).astype(np.int32)[:,:,::-1], rpc=True)
        # 去归一化: pu4c.det3d.app.image_viewer(data=(np.transpose(x[3].cpu().numpy(), (1, 2, 0)) + np.array([103.530, 116.280, 123.675])).astype(np.int32)[:,:,::-1], rpc=True)
        if np.random.rand() > self.prob or not self.training:
            return x
        n,c,h,w = x.size()
        x = x.view(-1,h,w)
        hh = int(1.5*h)
        ww = int(1.5*w)
        # d: 条幅宽度
        d = np.random.randint(2, h)
        # l: 掩盖宽度，ratio=0.5 为在条幅中掩盖的比率，并限制掩盖宽度小于条幅宽度
        self.l = min(max(int(d*self.ratio+0.5),1),d-1)
        # 1.5 倍尺度的掩膜
        mask = np.ones((hh, ww), np.float32)
        # 掩盖起点相对于条幅起点的位置偏移
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            # 图像高度方向上执行掩盖，高度方向上根据条幅宽度划分为多个条幅，计算掩盖条幅的起始高度和终止高度并令其掩膜值为 0
            for i in range(hh//d):
                s = d*i + st_h
                t = min(s+self.l, hh)
                mask[s:t,:] *= 0
        if self.use_w:
            # 图像宽度方向上同理
            for i in range(ww//d):
                s = d*i + st_w
                t = min(s+self.l, ww)
                mask[:,s:t] *= 0
       
        # 掩膜旋转，这里配置参数为 1 随机数只能取到 0 即不旋转
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        # mask = np.asarray(mask)
        mask = np.array(mask)
        # 1.5 倍掩膜中心裁剪到 1 倍即原图尺寸
        mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]

        mask = torch.from_numpy(mask).to(x.dtype).cuda()
        if self.mode == 1:
            # 执行，取反即只保留掩盖条幅部分
            mask = 1-mask
        mask = mask.expand_as(x)
        if self.offset:
            # 不执行，掩膜偏移
            offset = torch.from_numpy(2 * (np.random.rand(h,w) - 0.5)).to(x.dtype).cuda()
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask 
        
        # import pdb; pdb.set_trace()
        
        # @tag vis 可视化掩膜后图像: pu4c.det3d.app.image_viewer(data=np.transpose(x.view(n,c,h,w)[3].cpu().numpy(), (1, 2, 0)).astype(np.int32)[:,:,::-1], rpc=True)
        return x.view(n,c,h,w)

if __name__ == '__main__':
    pass