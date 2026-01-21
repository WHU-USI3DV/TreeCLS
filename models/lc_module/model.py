import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
# from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Union
import copy
from models import open_clip
import numpy as np

def independent_multilayer_cam(feature_dict: dict, logits_dict: dict, target_layers: list, is_training: bool):
    """
    分别为每一层计算其对应的 CAM。
    Layer1 Logits -> Layer1 Features
    Layer2 Logits -> Layer2 Features
    ...
    """
    masks = {}

    # 必须开启梯度环境
    with torch.enable_grad():
        for layer_name in target_layers:
            # 1. 获取成对的 Feature 和 Logit
            feat = feature_dict.get(layer_name)
            logit = logits_dict.get(layer_name)

            # 检查有效性
            if feat is None or logit is None:
                continue
            if not feat.requires_grad:
                continue

            # 2. 确定目标类别 (Target Class)
            # logit shape: [B, S, NumClasses] (基于你 WeaklySelector 的输出)
            # 我们需要先聚合空间维度，找到预测置信度最高的那个类
            probs = F.softmax(logit, dim=-1)      # [B, S, C]
            avg_probs = probs.mean(dim=1)         # [B, C] 全局平均池化
            target_ids = avg_probs.argmax(dim=-1) # [B] 预测的类别ID

            # 3. 构建 Target Score (标量)
            # 取出目标类别的分数，并对所有空间位置求和
            # logit: [B, S, C] -> gather -> [B, S, 1] -> sum -> [B] -> sum -> scalar
            target_score = logit.gather(2, target_ids.view(-1, 1, 1).expand(-1, logit.size(1), 1)).sum()

            # 4. 计算梯度
            # 这里的 inputs 只有当前的 feat，不会干扰其他层
            grads = torch.autograd.grad(
                outputs=target_score,
                inputs=feat,
                grad_outputs=None,
                retain_graph=is_training, # 必须为 True，因为可能有共享的计算图路径
                create_graph=False,
                allow_unused=True
            )[0]

            if grads is None:
                continue

            # 情况 A: 4D Tensor [B, C, H, W] (例如 ResNet原始特征)
            if len(feat.shape) == 4:
                # 对 H, W 求平均 (GAP) -> [B, C, 1, 1]
                weights = torch.mean(grads, dim=(2, 3), keepdim=True)
                # 加权求和 -> [B, H, W]
                cam = torch.sum(weights * feat, dim=1)
                # 展平以便归一化 -> [B, H*W]
                cam = cam.view(cam.shape[0], -1)

            # 情况 B: 3D Tensor [B, S, C] (例如 Transformer 或 展平后的特征)
            # 假设 shape 是 [Batch, Spatial, Channel]
            elif len(feat.shape) == 3:
                # 对 Spatial 维度求平均 -> [B, 1, C]
                # 注意：这里假设 dim=1 是 Spatial (序列长度)，dim=2 是 Channel
                # 如果你的数据是 [B, C, S]，请改为 dim=2
                weights = torch.mean(grads, dim=1, keepdim=True)

                # 加权求和 (Channel维相乘并求和) -> [B, S]
                cam = torch.sum(weights * feat, dim=-1)
            else:
                print(f"Skipping layer {layer_name}: unsupported shape {feat.shape}")
                continue

            # 6. 后处理 (Soft Version)
            cam = F.relu(cam)
            cam = cam.view(cam.shape[0], -1) # Flatten for normalization [B, H*W]
            min_v = cam.min(dim=1, keepdim=True)[0]
            max_v = cam.max(dim=1, keepdim=True)[0]
            cam = (cam - min_v) / (max_v - min_v + 1e-8)
            # cam = (cam > cam.quantile(0.5, dim=1, keepdim=True)).float()
            masks[layer_name] = cam.detach()

    return masks


def multi_layer_grad_cam(feature_dict: dict, logits: torch.Tensor, target_layers: list, is_training: bool):
    """
    一次性计算多个层的 Grad-CAM，避免反复构建/销毁计算图。

    Args:
        feature_dict: 包含所有特征的字典
        logits: 最终输出
        target_layers: 需要计算 mask 的层名列表，如 ['layer1', 'layer2'...]
        is_training: 是否训练模式
    """
    # 0. 准备 valid inputs
    valid_inputs = []
    valid_layers = []

    # 筛选出 requires_grad=True 的特征层
    for layer in target_layers:
        feat = feature_dict.get(layer)
        if feat is not None and feat.requires_grad:
            valid_inputs.append(feat)
            valid_layers.append(layer)

    if not valid_inputs:
        return {}

    masks = {}

    # 1. 强制梯度环境
    with torch.enable_grad():
        # 2. 准备 Target Score
        probs = F.softmax(logits, dim=-1)
        target_ids = probs.argmax(dim=-1)
        target_score = logits.gather(1, target_ids.unsqueeze(-1)).sum()

        # 3. 【核心修改】一次性对列表求导
        # inputs 是一个列表，grads 返回的也是一个对应顺序的梯度列表
        # 在测试时(is_training=False)，retain_graph=False。
        # 因为这是一次性计算，算完立刻销毁图是安全的，而且是最省显存的！
        grads_list = torch.autograd.grad(
            outputs=target_score,
            inputs=valid_inputs,
            grad_outputs=None,
            retain_graph=is_training, # 训练时保留(给loss用)，测试时不保留(用完即弃)
            create_graph=False,
            allow_unused=True
        )

        # 4. 遍历结果生成 Mask
        for i, feat in enumerate(valid_inputs):
            grad = grads_list[i]
            if grad is None: continue

            layer_name = valid_layers[i]

            # 根据维度计算 CAM
            if len(feat.shape) == 4: # [B, C, H, W]
                weights = torch.mean(grad, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * feat, dim=1)
                cam = cam.view(cam.shape[0], -1)
            elif len(feat.shape) == 3: # [B, S, C] or [B, C, S]
                if feat.shape[-1] == grad.shape[-1]:
                    weights = torch.mean(grad, dim=1, keepdim=True)
                    cam = torch.sum(weights * feat, dim=-1)
                else:
                    weights = torch.mean(grad, dim=2, keepdim=True)
                    cam = torch.sum(weights * feat, dim=1)

            # 后处理
            cam = F.relu(cam)
            min_v = cam.min(dim=1, keepdim=True)[0]
            max_v = cam.max(dim=1, keepdim=True)[0]
            cam = (cam - min_v) / (max_v - min_v + 1e-8)
            cam = (cam > cam.quantile(0.5, dim=1, keepdim=True)).float()
            # 存入结果
            masks[layer_name] = cam.detach()

    return masks



def convert_normalize(tensor):
    # 定义新旧归一化参数
    old_mean = [0.485, 0.456, 0.406]
    old_std = [0.229, 0.224, 0.225]
    new_mean = [0.48145466, 0.4578275, 0.40821073]
    new_std = [0.26862954, 0.26130258, 0.27577711]

    # 获取输入tensor的设备和数据类型
    device = tensor.device
    dtype = tensor.dtype

    # 转换为对应设备的tensor并调整维度 (1, 3, 1, 1)
    old_mean = torch.tensor(old_mean, device=device, dtype=dtype).view(1, 3, 1, 1)
    old_std = torch.tensor(old_std, device=device, dtype=dtype).view(1, 3, 1, 1)
    new_mean = torch.tensor(new_mean, device=device, dtype=dtype).view(1, 3, 1, 1)
    new_std = torch.tensor(new_std, device=device, dtype=dtype).view(1, 3, 1, 1)

    # 反归一化得到原始像素值
    original = tensor * old_std + old_mean

    # 使用新参数重新归一化
    new_tensor = (original - new_mean) / new_std

    return new_tensor

## Stage 2 ##

# Instance whitening
class InstanceWhitening(nn.Module):

    def __init__(self, dim):
        super(InstanceWhitening, self).__init__()
        self.instance_standardization = nn.InstanceNorm2d(dim, affine=False)

    def forward(self, x):

        x = self.instance_standardization(x)
        w = x

        return x, w


def instance_whitening_loss(f_map, eye, mask_matrix, margin, num_remove_cov):
    f_cor, B = get_covariance_matrix(f_map, eye=eye)
    f_cor_masked = f_cor * mask_matrix

    off_diag_sum = torch.sum(torch.abs(f_cor_masked), dim=(1,2), keepdim=True) - margin # B X 1 X 1
    loss = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0) # B X 1 X 1
    loss = torch.sum(loss) / B

    return loss


def get_covariance_matrix(f_map, eye=None):
    eps = 1e-5
    B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
    HW = H * W
    if eye is None:
        eye = torch.eye(C).cuda()
    f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW-1) + (eps * eye)  # C X C / HW

    return f_cor, B

# Semantic concept modeling for swin
class SCMModule(nn.Module):

    def __init__(self,
                 outs: dict,
                 fine_num_classes: int,
                 coarse_num_classes: int,
                 mid_feature_num: int):

        super(SCMModule, self).__init__()

        if len(outs['layer4'].shape) == 4:
            input_feature_num = outs['layer4'].shape[-1] * outs['layer4'].shape[-2]
        elif len(outs['layer4'].shape) == 3:
            input_feature_num = outs['layer4'].shape[1]
        self.branch1_linear1 = nn.Sequential(nn.Linear(input_feature_num, mid_feature_num), nn.ReLU()) # swin-144
        self.branch1_linear2 = nn.Linear(mid_feature_num, coarse_num_classes)
        self.branch1_iw = InstanceWhitening(coarse_num_classes)

        self.branch2_linear1 = nn.Sequential(nn.Linear(input_feature_num, mid_feature_num), nn.ReLU()) # swin-144
        self.branch2_linear2 = nn.Linear(mid_feature_num, fine_num_classes)
        self.branch21_linear = nn.Linear(fine_num_classes, coarse_num_classes)
        self.branch21_iw = InstanceWhitening(coarse_num_classes)
        self.branch22_linear = nn.Linear(fine_num_classes, fine_num_classes)
        self.branch22_iw = InstanceWhitening(fine_num_classes)

        self.constraint = nn.MSELoss()

    def forward(self, x):
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x = x.view((B, C, H*W))
        else:
            x = x.transpose(1, 2).contiguous()
        branch1 = self.branch1_linear1(x)
        branch1 = self.branch1_linear2(branch1)
        branch1 = branch1.transpose(1, 2).contiguous()
        branch1 = branch1.unsqueeze(3)
        branch1, _ = self.branch1_iw(branch1)
        branch1 = branch1.squeeze(3)
        branch2 = self.branch2_linear1(x)
        branch2 = self.branch2_linear2(branch2)
        branch21 = self.branch21_linear(branch2)
        branch21 = branch21.transpose(1, 2).contiguous()
        branch21 = branch21.unsqueeze(3)
        branch21, _ = self.branch21_iw(branch21)
        branch21 = branch21.squeeze(3)
        branch22 = self.branch22_linear(branch2)
        branch22 = branch22.transpose(1, 2).contiguous()
        branch22 = branch22.unsqueeze(3)
        output, _ = self.branch22_iw(branch22)
        output = output.squeeze(3)
        constraint = self.constraint(branch1, branch21)

        return output, constraint

# Semantic concept embedding
class SCEModule(nn.Module):

    def __init__(self):

        super(SCEModule, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.linear_b1 = nn.Linear(3, 1)
        self.linear_b2 = nn.Linear(3, 1)
        self.linear_b3 = nn.Linear(3, 1)
        self.linear_b4 = nn.Linear(3, 1)

    def forward(self, x, out1):
        output = {}
        mask1 = self.sigmoid(out1.sum(dim=1))
        mask1 = mask1.unsqueeze(1)

        for name in x:
            if len(x[name].shape) == 4:
                B, C, H, W = x[name].shape
                x[name] = x[name].view((B, C, H*W))
                x[name] = x[name].transpose(1, 2).contiguous()

        g_b1 = self.relu(torch.mul(x['layer1'], mask1))
        g_b2 = self.relu(torch.mul(x['layer2'], mask1))
        g_b3 = self.relu(torch.mul(x['layer3'], mask1))
        g_b4 = self.relu(torch.mul(x['layer4'], mask1))

        mask_avg_b1 = torch.mean(x['layer1'], dim=1, keepdim=True)
        mask_max_b1, _ = torch.max(x['layer1'], dim=1, keepdim=True)
        mask_avg_b2 = torch.mean(x['layer2'], dim=1, keepdim=True)
        mask_max_b2, _ = torch.max(x['layer2'], dim=1, keepdim=True)
        mask_avg_b3 = torch.mean(x['layer3'], dim=1, keepdim=True)
        mask_max_b3, _ = torch.max(x['layer3'], dim=1, keepdim=True)
        mask_avg_b4 = torch.mean(x['layer4'], dim=1, keepdim=True)
        mask_max_b4, _ = torch.max(x['layer4'], dim=1, keepdim=True)

        mask2_b1 = torch.cat([mask_max_b1, mask_avg_b1, mask1], dim=1).transpose(1, 2).contiguous()
        mask2_b2 = torch.cat([mask_max_b2, mask_avg_b2, mask1], dim=1).transpose(1, 2).contiguous()
        mask2_b3 = torch.cat([mask_max_b3, mask_avg_b3, mask1], dim=1).transpose(1, 2).contiguous()
        mask2_b4 = torch.cat([mask_max_b4, mask_avg_b4, mask1], dim=1).transpose(1, 2).contiguous()

        mask3_b1 = self.sigmoid(self.linear_b1(mask2_b1)).transpose(1, 2).contiguous()
        mask3_b2 = self.sigmoid(self.linear_b2(mask2_b2)).transpose(1, 2).contiguous()
        mask3_b3 = self.sigmoid(self.linear_b3(mask2_b3)).transpose(1, 2).contiguous()
        mask3_b4 = self.sigmoid(self.linear_b4(mask2_b4)).transpose(1, 2).contiguous()

        output['layer1'] = torch.mul(g_b1, mask3_b1)
        output['layer2'] = torch.mul(g_b2, mask3_b2)
        output['layer3'] = torch.mul(g_b3, mask3_b3)
        output['layer4'] = torch.mul(g_b4, mask3_b4)

        return output

# Semantic constraint combiner
class SCCombiner(nn.Module):

    def __init__(self,
                 outs: dict,
                 total_num_selects: int,
                 num_classes: int,
                 inputs: Union[dict, None] = None,
                 proj_size: Union[int, None] = None,
                 fpn_size: Union[int, None] = None):
        """
        If building backbone without FPN, set fpn_size to None and MUST give
        'inputs' and 'proj_size', the reason of these setting is to constrain the
        dimension of graph convolutional network input.
        """
        super(SCCombiner, self).__init__()

        assert inputs is not None or fpn_size is not None, \
            "To build GCN combiner, you must give one features dimension."

        ### auto-proj
        self.fpn_size = fpn_size
        # branch_size = [2048, 512, 128, 32]
        # branch_size = [2304, 576, 144, 144]
        if len(outs['layer1'].shape) == 4:
            branch_size = [outs['layer1'].shape[-1]**2, outs['layer2'].shape[-1]**2, outs['layer3'].shape[-1]**2, outs['layer4'].shape[-1]**2]
        else:
            branch_size = [outs['layer1'].shape[-2], outs['layer2'].shape[-2], outs['layer3'].shape[-2], outs['layer4'].shape[-2]]

        if fpn_size is None:
            for name in inputs:
                fs_size = inputs[name].size()
                if len(fs_size) == 3:
                    in_size = fs_size[2]
                elif len(fs_size) == 4:
                    in_size = fs_size[1]
                else:
                    raise ValusError("The size of output dimension of previous must be 3 or 4.")
                m = nn.Sequential(
                    nn.Linear(in_size, proj_size),
                    nn.ReLU(),
                    nn.Linear(proj_size, proj_size)
                )
                self.add_module("proj_"+name, m)
            self.proj_size = proj_size
        else:
            self.proj_size = fpn_size

        ### merge information
        self.pool0_b1 = nn.Sequential(nn.Linear(branch_size[0], 1), nn.ReLU())
        self.pool0_b2 = nn.Sequential(nn.Linear(branch_size[1], 1), nn.ReLU())
        self.pool0_b3 = nn.Sequential(nn.Linear(branch_size[2], 1), nn.ReLU())
        self.pool0_b4 = nn.Sequential(nn.Linear(branch_size[3], 1), nn.ReLU())

        self.pool1_b1 = nn.Linear(self.proj_size, num_classes)
        self.pool1_b2 = nn.Linear(self.proj_size, num_classes)
        self.pool1_b3 = nn.Linear(self.proj_size, num_classes)
        self.pool1_b4 = nn.Linear(self.proj_size, num_classes)

        self.norm = nn.Sigmoid()
        self.constraint = nn.MSELoss()

    def forward(self, x):
        """
        """
        hs = []
        for name in x:
            if self.fpn_size is None:
                hs.append(getattr(self, "proj_"+name)(x[name]))
            else:
                hs.append(x[name])
        for i in range(len(hs)):
            hs[i] = hs[i].transpose(1, 2).contiguous()
        hs[0] = self.pool0_b1(hs[0])
        hs[1] = self.pool0_b2(hs[1])
        hs[2] = self.pool0_b3(hs[2])
        hs[3] = self.pool0_b4(hs[3])
        for i in range(len(hs)):
            hs[i] = hs[i].flatten(1)
        hs[0] = self.pool1_b1(hs[0])
        hs[1] = self.pool1_b2(hs[1])
        hs[2] = self.pool1_b3(hs[2])
        hs[3] = self.pool1_b4(hs[3])

        ## Gram matrix for constraint
        if len(hs[0].shape) == 2:
            gram4 = torch.matmul(hs[0].transpose(0, 1).contiguous(), hs[0])
            gram3 = torch.matmul(hs[1].transpose(0, 1).contiguous(), hs[1])
            gram2 = torch.matmul(hs[2].transpose(0, 1).contiguous(), hs[2])
            gram1 = torch.matmul(hs[3].transpose(0, 1).contiguous(), hs[3])
        elif len(hs[0].shape) == 3:
            gram4 = torch.matmul(hs[0].transpose(1, 2).contiguous(), hs[0])
            gram3 = torch.matmul(hs[1].transpose(1, 2).contiguous(), hs[1])
            gram2 = torch.matmul(hs[2].transpose(1, 2).contiguous(), hs[2])
            gram1 = torch.matmul(hs[3].transpose(1, 2).contiguous(), hs[3])
        constraint_loss = torch.abs(self.constraint(gram2, gram1))
        constraint_loss += torch.abs(self.constraint(gram3, gram1))
        constraint_loss += torch.abs(self.constraint(gram4, gram1))

        final_h = hs[0] + hs[1] + hs[2] + hs[3]
        final_h = self.norm(final_h)

        return final_h, constraint_loss



## Origin
class NEWCombiner(nn.Module):

    def __init__(self,
                 total_num_selects: int,
                 num_classes: int,
                 inputs: Union[dict, None] = None,
                 proj_size: Union[int, None] = None,
                 fpn_size: Union[int, None] = None):
        """
        If building backbone without FPN, set fpn_size to None and MUST give
        'inputs' and 'proj_size', the reason of these setting is to constrain the
        dimension of graph convolutional network input.
        """
        super(NEWCombiner, self).__init__()

        assert inputs is not None or fpn_size is not None, \
            "To build GCN combiner, you must give one features dimension."

        ### auto-proj
        self.fpn_size = fpn_size
        branch_size = [2048, 512, 128, 32]
        if fpn_size is None:
            for name in inputs:
                fs_size = inputs[name].size()
                if len(fs_size) == 3:
                    in_size = fs_size[2]
                elif len(fs_size) == 4:
                    in_size = fs_size[1]
                else:
                    raise ValusError("The size of output dimension of previous must be 3 or 4.")
                m = nn.Sequential(
                    nn.Linear(in_size, proj_size),
                    nn.ReLU(),
                    nn.Linear(proj_size, proj_size)
                )
                self.add_module("proj_"+name, m)
            self.proj_size = proj_size
        else:
            self.proj_size = fpn_size

        ### merge information
        self.pool0_b1 = nn.Sequential(nn.Linear(branch_size[0], 1), nn.ReLU())
        self.pool0_b2 = nn.Sequential(nn.Linear(branch_size[1], 1), nn.ReLU())
        self.pool0_b3 = nn.Sequential(nn.Linear(branch_size[2], 1), nn.ReLU())
        self.pool0_b4 = nn.Sequential(nn.Linear(branch_size[3], 1), nn.ReLU())

        self.pool1_b1 = nn.Linear(self.proj_size, num_classes)
        self.pool1_b2 = nn.Linear(self.proj_size, num_classes)
        self.pool1_b3 = nn.Linear(self.proj_size, num_classes)
        self.pool1_b4 = nn.Linear(self.proj_size, num_classes)

        self.norm = nn.Sigmoid()

    def forward(self, x):
        """
        """
        hs = []
        for name in x:
            if self.fpn_size is None:
                hs.append(getattr(self, "proj_"+name)(x[name]))
            else:
                hs.append(x[name])
        for i in range(len(hs)):
            hs[i] = hs[i].transpose(1, 2).contiguous()
        hs[0] = self.pool0_b1(hs[0])
        hs[1] = self.pool0_b2(hs[1])
        hs[2] = self.pool0_b3(hs[2])
        hs[3] = self.pool0_b4(hs[3])
        for i in range(len(hs)):
            hs[i] = hs[i].flatten(1)
        hs[0] = self.pool1_b1(hs[0])
        hs[1] = self.pool1_b2(hs[1])
        hs[2] = self.pool1_b3(hs[2])
        hs[3] = self.pool1_b4(hs[3])
        final_h = hs[0] + hs[1] + hs[2] + hs[3]
        final_h = self.norm(final_h)

        return final_h

class GCNCombiner(nn.Module):

    def __init__(self,
                 total_num_selects: int,
                 num_classes: int,
                 inputs: Union[dict, None] = None,
                 proj_size: Union[int, None] = None,
                 fpn_size: Union[int, None] = None):
        """
        If building backbone without FPN, set fpn_size to None and MUST give
        'inputs' and 'proj_size', the reason of these setting is to constrain the
        dimension of graph convolutional network input.
        """
        super(GCNCombiner, self).__init__()

        assert inputs is not None or fpn_size is not None, \
            "To build GCN combiner, you must give one features dimension."

        ### auto-proj
        self.fpn_size = fpn_size
        if fpn_size is None:
            for name in inputs:
                fs_size = inputs[name].size()
                if len(fs_size) == 3:
                    in_size = fs_size[2]
                elif len(fs_size) == 4:
                    in_size = fs_size[1]
                else:
                    raise ValusError("The size of output dimension of previous must be 3 or 4.")
                m = nn.Sequential(
                    nn.Linear(in_size, proj_size),
                    nn.ReLU(),
                    nn.Linear(proj_size, proj_size)
                )
                self.add_module("proj_"+name, m)
            self.proj_size = proj_size
        else:
            self.proj_size = fpn_size

        ### build one layer structure (with adaptive module)
        num_joints = total_num_selects // 32

        self.param_pool0 = nn.Linear(total_num_selects, num_joints)

        A = torch.eye(num_joints)/100 + 1/100
        self.adj1 = nn.Parameter(copy.deepcopy(A))
        self.conv1 = nn.Conv1d(self.proj_size, self.proj_size, 1)
        self.batch_norm1 = nn.BatchNorm1d(self.proj_size)

        self.conv_q1 = nn.Conv1d(self.proj_size, self.proj_size//4, 1)
        self.conv_k1 = nn.Conv1d(self.proj_size, self.proj_size//4, 1)
        self.alpha1 = nn.Parameter(torch.zeros(1))

        ### merge information
        self.param_pool1 = nn.Linear(num_joints, 1)

        #### class predict
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.proj_size, num_classes)

        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        """
        hs = []
        for name in x:
            if self.fpn_size is None:
                hs.append(getattr(self, "proj_"+name)(x[name]))
            else:
                hs.append(x[name])
        hs = torch.cat(hs, dim=1).transpose(1, 2).contiguous() # B, S', C --> B, C, S
        hs = self.param_pool0(hs)
        ### adaptive adjacency
        q1 = self.conv_q1(hs).mean(1)
        k1 = self.conv_k1(hs).mean(1)
        A1 = self.tanh(q1.unsqueeze(-1) - k1.unsqueeze(1))
        A1 = self.adj1 + A1 * self.alpha1
        ### graph convolution
        hs = self.conv1(hs)
        hs = torch.matmul(hs, A1)
        hs = self.batch_norm1(hs)
        ### predict
        hs = self.param_pool1(hs)
        hs = self.dropout(hs)
        hs = hs.flatten(1)
        hs = self.classifier(hs)

        return hs

class WeaklySelector(nn.Module):

    def __init__(self, inputs: dict, num_classes: int, num_select: dict, fpn_size: Union[int, None] = None):
        """
        inputs: dictionary contain torch.Tensors, which comes from backbone
                [Tensor1(hidden feature1), Tensor2(hidden feature2)...]
                Please note that if len(features.size) equal to 3, the order of dimension must be [B,S,C],
                S mean the spatial domain, and if len(features.size) equal to 4, the order must be [B,C,H,W]

        """
        super(WeaklySelector, self).__init__()

        self.num_select = num_select

        self.fpn_size = fpn_size
        ### build classifier
        if self.fpn_size is None:
            self.num_classes = num_classes
            for name in inputs:
                fs_size = inputs[name].size()
                if len(fs_size) == 3:
                    in_size = fs_size[2]
                elif len(fs_size) == 4:
                    in_size = fs_size[1]
                m = nn.Linear(in_size, num_classes)
                self.add_module("classifier_l_"+name, m)

    def get_mask(self, ranks, num_select, shape):
        mask = torch.zeros(shape)
        mask[ranks[:num_select]] = 1
        return mask

    def hard_threshold_mask(self, x, threshold=0.5):
        """
        实现精确的硬阈值二值化，同时保持梯度回传

        前向传播:
            x < 0.5 → 0
            x ≥ 0.5 → 1

        反向传播:
            梯度直接通过(就像没有进行二值化操作一样)
        """
        # 前向：硬阈值二值化
        mask = (x >= threshold).float()

        # 反向：直通估计器(STE) - 梯度直接通过
        # return mask + x - x.detach()  # 等价于 mask，但保留了x的梯度路径
        return mask.detach()

    def forward(self, x, logits=None):
        """
        x :
            dictionary contain the features maps which
            come from your choosen layers.
            size must be [B, HxW, C] ([B, S, C]) or [B, C, H, W].
            [B,C,H,W] will be transpose to [B, HxW, C] automatically.
        """
        if self.fpn_size is None:
            logits = {}
        selections = {}
        preds_1 = {}
        preds_0 = {}
        for name in x:
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                x[name] = x[name].view(B, C, H*W).permute(0, 2, 1).contiguous()
            C = x[name].size(-1)
            if self.fpn_size is None:
                logits[name] = getattr(self, "classifier_l_"+name)(x[name])

            probs = torch.softmax(logits[name], dim=-1)
            selections[name] = []
            preds_1[name] = []
            preds_0[name] = []

            weights = []
            num_select = self.num_select[name]
            for bi in range(logits[name].size(0)):
                max_ids, _ = torch.max(probs[bi], dim=-1)
                confs, ranks = torch.sort(max_ids, descending=True)
                sf = x[name][bi][ranks[:num_select]]
                nf = x[name][bi][ranks[num_select:]]  # calculate

                threshold = confs[num_select-1]
                weights.append(self.hard_threshold_mask(probs[bi, :, max_ids], threshold))

                selections[name].append(sf) # [num_selected, C]
                preds_1[name].append(logits[name][bi][ranks[:num_select]])
                preds_0[name].append(logits[name][bi][ranks[num_select:]])

            selections[name] = torch.stack(selections[name])
            preds_1[name] = torch.stack(preds_1[name])
            preds_0[name] = torch.stack(preds_0[name])
            logits["weight_"+name.replace('layer','')] = torch.stack(weights).to(x[name].device)

        return selections


class FPN(nn.Module):

    def __init__(self, inputs: dict, fpn_size: int, proj_type: str, upsample_type: str):
        """
        inputs : dictionary contains torch.Tensor
                 which comes from backbone output
        fpn_size: integer, fpn
        proj_type:
            in ["Conv", "Linear"]
        upsample_type:
            in ["Bilinear", "Conv", "Fc"]
            for convolution neural network (e.g. ResNet, EfficientNet), recommand 'Bilinear'.
            for Vit, "Fc". and Swin-T, "Conv"
        """
        super(FPN, self).__init__()
        assert proj_type in ["Conv", "Linear"], \
            "FPN projection type {} were not support yet, please choose type 'Conv' or 'Linear'".format(proj_type)
        assert upsample_type in ["Bilinear", "Conv"], \
            "FPN upsample type {} were not support yet, please choose type 'Bilinear' or 'Conv'".format(proj_type)

        self.fpn_size = fpn_size
        self.upsample_type = upsample_type
        inp_names = [name for name in inputs]

        for i, node_name in enumerate(inputs):
            ### projection module
            if proj_type == "Conv":
                m = nn.Sequential(
                    nn.Conv2d(inputs[node_name].size(1), inputs[node_name].size(1), 1),
                    nn.ReLU(),
                    nn.Conv2d(inputs[node_name].size(1), fpn_size, 1)
                )
            elif proj_type == "Linear":
                m = nn.Sequential(
                    nn.Linear(inputs[node_name].size(-1), inputs[node_name].size(-1)),
                    nn.ReLU(),
                    nn.Linear(inputs[node_name].size(-1), fpn_size),
                )
            self.add_module("Proj_"+node_name, m)

            ### upsample module
            if upsample_type == "Conv" and i != 0:
                assert len(inputs[node_name].size()) == 3 # B, S, C
                in_dim = inputs[node_name].size(1)
                out_dim = inputs[inp_names[i-1]].size(1)
                if in_dim != out_dim:
                    m = nn.Conv1d(in_dim, out_dim, 1) # for spatial domain
                else:
                    m = nn.Identity()
                self.add_module("Up_"+node_name, m)

        if upsample_type == "Bilinear":
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def upsample_add(self, x0: torch.Tensor, x1: torch.Tensor, x1_name: str):
        """
        return Upsample(x1) + x1
        """
        if self.upsample_type == "Bilinear":
            if x1.size(-1) != x0.size(-1):
                x1 = self.upsample(x1)
        else:
            x1 = getattr(self, "Up_"+x1_name)(x1)
        return x1 + x0

    def forward(self, x):
        """
        x : dictionary
            {
                "node_name1": feature1,
                "node_name2": feature2, ...
            }
        """
        ### project to same dimension
        hs = []
        for i, name in enumerate(x):
            x[name] = getattr(self, "Proj_"+name)(x[name])
            hs.append(name)

        for i in range(len(hs)-1, 0, -1):
            x1_name = hs[i]
            x0_name = hs[i-1]
            x[x0_name] = self.upsample_add(x[x0_name],
                                           x[x1_name],
                                           x1_name)
        return x


class LCModel(nn.Module):

    def __init__(self,
                 backbone: torch.nn.Module,
                 return_nodes: Union[dict, None],
                 img_size: int,
                 use_fpn: bool,
                 fpn_size: Union[int, None],
                 proj_type: str,
                 upsample_type: str,
                 use_selection: bool,
                 num_classes: int,
                 num_selects: dict,
                 use_combiner: bool,
                 comb_proj_size: Union[int, None],
                 coarse_classes: Union[int, None] = None,
                 add_linear: bool = False,
                 feature_type: Union[str, None] = None,
                 add_loss: bool = False,
                 only_loss: bool = False,
                 no_mask: bool = False,
                 use_embedding: bool = False,
                 pretrained_path: Union[str, None] = None,
                 use_cam: bool = False,
                 num_bins: int = 8,
                 fuse_type: int = 0,
                 ):
        """
        * backbone:
            torch.nn.Module class (recommand pretrained on ImageNet or IG-3.5B-17k(provided by FAIR))
        * return_nodes:
            e.g.
            return_nodes = {
                # node_name: user-specified key for output dict
                'layer1.2.relu_2': 'layer1',
                'layer2.3.relu_2': 'layer2',
                'layer3.5.relu_2': 'layer3',
                'layer4.2.relu_2': 'layer4',
            } # you can see the example on https://pytorch.org/vision/main/feature_extraction.html
            !!! if using 'Swin-Transformer', please set return_nodes to None
            !!! and please set use_fpn to True
        * feat_sizes:
            tuple or list contain features map size of each layers.
            ((C, H, W)). e.g. ((1024, 14, 14), (2048, 7, 7))
        * use_fpn:
            boolean, use features pyramid network or not
        * fpn_size:
            integer, features pyramid network projection dimension
        * num_selects:
            num_selects = {
                # match user-specified in return_nodes
                "layer1": 2048,
                "layer2": 512,
                "layer3": 128,
                "layer4": 32,
            }

        Note: after selector module (WeaklySelector) , the feature map's size is [B, S', C] which
        contained by 'logits' or 'selections' dictionary (S' is selection number, different layer
        could be different).
        """
        super(LCModel, self).__init__()
        ### = = = = = Backbone = = = = =
        self.return_nodes = return_nodes
        if return_nodes is not None:
            self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)
        else:
            self.backbone = backbone

        ### get hidden feartues size
        rand_in = torch.randn(1, 3, img_size, img_size)
        # outs = self.backbone(rand_in)
        out = self.backbone(rand_in)
        if type(out) == list:
            outs = {}
            outs['layer1'] = out[0]
            outs['layer2'] = out[1]
            outs['layer3'] = out[2]
            outs['layer4'] = out[3]
        else:
            outs = out

        ### just original backbone
        if not use_fpn and (not use_selection and not use_combiner):
            for name in outs:
                fs_size = outs[name].size()
                if len(fs_size) == 3:
                    out_size = fs_size[-1]
                elif len(fs_size) == 4:
                    out_size = fs_size[1]
                else:
                    raise ValusError("The size of output dimension of previous must be 3 or 4.")
            self.classifier = nn.Linear(out_size, num_classes)
        ### = = = = = FPN = = = = =
        self.use_fpn = use_fpn
        if self.use_fpn:
            self.fpn = FPN(outs, fpn_size, proj_type, upsample_type)
            self.build_fpn_classifier(outs, fpn_size, num_classes)

        self.fpn_size = fpn_size

        ### = = = = = Selector = = = = =
        self.use_selection = use_selection
        if self.use_selection:
            w_fpn_size = self.fpn_size if self.use_fpn else None # if not using fpn, build classifier in weakly selector
            self.selector = WeaklySelector(outs, num_classes, num_selects, w_fpn_size)

        ### = = = = = Combiner = = = = =
        self.use_combiner = use_combiner
        if self.use_combiner:
            assert self.use_selection, "Please use selection module before combiner"
            if self.use_fpn:
                gcn_inputs, gcn_proj_size = None, None
            else:
                gcn_inputs, gcn_proj_size = outs, comb_proj_size # redundant, fix in future
                gcn_proj_size = 1536
            total_num_selects = sum([num_selects[name] for name in num_selects]) # sum
            # self.combiner = GCNCombiner(total_num_selects, num_classes, gcn_inputs, gcn_proj_size, self.fpn_size)
            self.combiner = NEWCombiner(total_num_selects, num_classes, gcn_inputs, gcn_proj_size, self.fpn_size)

        ### Stage2
        self.scm_module = SCMModule(outs, num_classes, 10, 64)
        self.sce_module = SCEModule()
        if self.use_combiner:
            self.combiner = SCCombiner(outs, total_num_selects, num_classes, gcn_inputs, gcn_proj_size, self.fpn_size)

        self.no_mask = no_mask
        self.use_embedding = use_embedding
        self.add_linear = add_linear
        self.fuse_type = fuse_type

        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path)
            self.load_state_dict(ckpt['model'], strict=True)
            for param in self.parameters():
                param.requires_grad = False
            del ckpt
            self.frozen_backbone = True
        else:
            self.frozen_backbone = False

        if self.add_linear:
            self.prior_model = PriorModel(self.no_mask, num_classes)
            self.text_pooling = nn.Linear(5, 1)
            self.classifier = nn.Linear(768, num_classes)

            if self.use_embedding and self.fuse_type == 0:
                self.adn = LogitsFusion(num_classes, num_bins)
            elif self.fuse_type == 1:
                self.logits_fuse_mlp = nn.Sequential(
                    nn.Linear(num_classes*2, num_classes),
                    nn.ReLU(),
                    nn.Linear(num_classes, num_classes)
                )
            elif self.fuse_type == 2:
                self.feature_fuse_mlp = nn.Sequential(
                    nn.Linear(768 + self.fpn_size, 512),
                    nn.ReLU(),
                    nn.Linear(512, num_classes)
                )
            elif self.fuse_type == 3:
                self.logits_proj = nn.Sequential(
                    nn.Linear(num_classes, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64)
                )
                self.logits_attn = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
                self.fc = nn.Sequential(
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes)
                )
            elif self.fuse_type == 4:
                self.text_proj = nn.Sequential(
                    nn.Linear(768, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64)
                )
                self.vision_proj = nn.Sequential(
                    nn.Linear(self.fpn_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64)
                )
                self.feature_attn = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
                self.fc2 = nn.Sequential(
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes)
                )

    def build_fpn_classifier(self, inputs: dict, fpn_size: int, num_classes: int):
        """
        Teh results of our experiments show that linear classifier in this case may cause some problem.
        """
        for name in inputs:
            m = nn.Sequential(
                    nn.Conv1d(fpn_size, fpn_size, 1),
                    nn.BatchNorm1d(fpn_size),
                    nn.ReLU(),
                    nn.Conv1d(fpn_size, num_classes, 1)
                )
            self.add_module("fpn_classifier_"+name, m)

    def forward_backbone(self, x):

        out = self.backbone(x)
        if type(out) == list:
            outs = {}
            outs['layer1'] = out[0]
            outs['layer2'] = out[1]
            outs['layer3'] = out[2]
            outs['layer4'] = out[3]
        else:
            outs = out

        # return self.backbone(x)
        return outs

    def fpn_predict(self, x: dict, logits: dict):
        """
        x: [B, C, H, W] or [B, S, C]
           [B, C, H, W] --> [B, H*W, C]
        """
        for name in x:
            ### predict on each features point
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                logit = x[name].view(B, C, H*W)
            elif len(x[name].size()) == 3:
                logit = x[name].transpose(1, 2).contiguous()
            logits[name] = getattr(self, "fpn_classifier_"+name)(logit)
            logits[name] = logits[name].transpose(1, 2).contiguous() # transpose

    def forward(self, x: torch.Tensor):

        logits = {}

        if self.add_linear:
            ori_img = convert_normalize(x)
            ori_img = F.interpolate(ori_img, size=(224, 224), mode='bicubic', align_corners=True)

            with torch.enable_grad():
                if self.frozen_backbone:
                    # 冻结时：先计算数值(不存图)，再 detach 出来手动设为起点
                    with torch.no_grad():
                        feat_dict = self.forward_backbone(x)
                    feat_dict = {k: v.detach().requires_grad_(True) for k, v in feat_dict.items()}
                else:
                    # 训练时：正常计算，保留中间梯度
                    feat_dict = self.forward_backbone(x)
                    for v in feat_dict.values(): v.retain_grad()
                if self.use_fpn:
                    feat_dict = self.fpn(feat_dict)
                    self.fpn_predict(feat_dict, logits)
                    for v in feat_dict.values():
                            if isinstance(v, torch.Tensor) and v.requires_grad:
                                v.retain_grad()

                target_layer_keys = ['layer1', 'layer2', 'layer3', 'layer4']
                generated_masks = independent_multilayer_cam(
                    feature_dict=feat_dict,
                    logits_dict=logits,
                    target_layers=target_layer_keys,
                    is_training=self.training
                )

                for full_key, mask in generated_masks.items():
                    # key处理: FPN1_layer1 -> weight_1
                    clean_key = full_key.replace("FPN1_", "").replace("layer", "weight_")
                    logits[clean_key] = mask

                # Semantic concept modeling
                output1, constraint1 = self.scm_module(feat_dict['layer4'])
                # Semantic concept embedding
                output2 = self.sce_module(feat_dict, output1)
                comb_outs, constraint2 = self.combiner(output2)

            self.prior_model.eval()
            with torch.no_grad():
                text_feature = self.prior_model(ori_img, logits)
            text_feature = text_feature.detach()
            text_feature = self.text_pooling(text_feature).squeeze(-1) # B, 768
            out = self.classifier(text_feature)
            logits['text_outs'] = out
            logits['visual_outs'] = comb_outs

            if self.use_embedding and self.fuse_type == 0:
                fused_logits, text_weight = self.adn(comb_outs, out)
            elif self.fuse_type == 1:
                fused_input = torch.cat([comb_outs, out], dim=-1)
                fused_logits = self.logits_fuse_mlp(fused_input)
                text_weight = torch.ones_like(fused_logits[:, :1])
            elif self.fuse_type == 2:
                fused_feature = torch.cat([text_feature, feat_dict['layer4'].mean(dim=1)], dim=-1)
                fused_logits = self.feature_fuse_mlp(fused_feature)
                text_weight = torch.ones_like(fused_logits[:, :1])
            elif self.fuse_type == 3:
                vision_proj = self.logits_proj(comb_outs).unsqueeze(1)  # B, 1, 64
                text_proj = self.logits_proj(out).unsqueeze(1)  # B, 1, 64
                fused_feat = self.logits_attn(vision_proj, text_proj, text_proj)[0] + vision_proj
                fused_logits = self.fc(fused_feat.squeeze(1))
                text_weight = torch.ones_like(fused_logits[:, :1])
            elif self.fuse_type == 4:
                text_proj = self.text_proj(text_feature).unsqueeze(1)  # B, 1, 64
                vision_proj = self.vision_proj(feat_dict['layer4'].mean(dim=1)).unsqueeze(1)  # B, 1, 64
                fused_feat = self.feature_attn(vision_proj, text_proj, text_proj)[0] + vision_proj
                fused_logits = self.fc2(fused_feat.squeeze(1))
                text_weight = torch.ones_like(fused_logits[:, :1])
            else:
                fused_logits = comb_outs + out
                text_weight = torch.ones_like(fused_logits[:, :1])

            logits['comb_outs'] = fused_logits
            logits['text_weight'] = text_weight.detach()
            logits['constraint1'] = constraint1
            logits['constraint2'] = constraint2
        else:
            x = self.forward_backbone(x)
            if self.use_fpn:
                x = self.fpn(x)
                self.fpn_predict(x, logits)
            output1, constraint1 = self.scm_module(x['layer4'])
            output2 = self.sce_module(x, output1)
            comb_outs, constraint2 = self.combiner(output2)
            logits['comb_outs'] = comb_outs
            logits['constraint1'] = constraint1
            logits['constraint2'] = constraint2

        if not self.training:
            for k, v in logits.items():
                if isinstance(v, torch.Tensor):
                    logits[k] = v.detach()
        return logits


class PriorModel(nn.Module):
    def __init__(self, no_mask=False, num_classes=102):
        super(PriorModel, self).__init__()
        self.no_mask = no_mask

        if num_classes == 100:
            ## 使用openclip的VIT-L模型
            self.model, _, self.tokenizer = open_clip.create_model_and_transforms(
                'ViT-L-14',
                pretrained='laion2b_s32b_b82k',  # 指定预训练权重来源
            )
        else:
            self.model, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
            self.tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')

        ## 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False

    def get_text_embedding(self, device):
        with torch.no_grad():
            text = self.tokenizer(self.class_text).to(device)
            text_features = self.model.encode_text(text, normalize=True)
        self.text_features = text_features

    @torch.no_grad()
    def forward(self, x, logits: dict):
        ## 102, 768
        B = x.shape[0]
        feature = []
        clip_feature = self.model.encode_image(x, normalize=True)[0]
        feature.append(clip_feature)
        if self.no_mask:
            feature = [clip_feature] * 5
        else:
            for i in range(4):
                name = f'weight_{i+1}'
                weight = logits[name]
                H, W = int(weight.shape[1]**0.5), int(weight.shape[1]**0.5)
                # for weight reshape
                weight = weight.view(B, H, W).unsqueeze(1) # B, 1, H, W
                weight = F.interpolate(weight, size=(16, 16), mode='bilinear', align_corners=True)
                weight = (weight > weight.quantile(0.5, dim=1, keepdim=True)).float()
                weight = weight.view(B, 1, 16*16).squeeze(1) # B, 256
                clip_feature = self.model.encode_mask_image(x, weight)[0]
                feature.append(clip_feature)

        feature = torch.stack(feature, dim=1) # B, 4, 768
        feature = feature.permute(0, 2, 1).contiguous() # B, 768, 4

        return feature

class LogitsFusion(nn.Module):
    def __init__(self, num_classes=102, num_bins=8):
        super(LogitsFusion, self).__init__()

        if num_classes == 200:
            self.class_counts = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30]

        elif num_classes == 102:
            self.class_counts = np.array([66, 86, 55, 52, 67, 35, 89, 94, 24, 29, 34, 60, 43, 114, 64, 83, 59, 42, 12, 72, 47, 105, 132, 52, 23, 24, 26, 16, 32, 33, 66, 83, 42, 53, 8, 137, 34, 43, 21, 96, 12, 56, 63, 52, 12, 40, 24, 72, 68, 24, 32, 52, 15, 21, 24, 74, 14, 4, 64, 40, 20, 48, 15, 32, 10, 56, 42, 50, 16, 18, 3, 16, 4, 2, 21, 4, 48, 4, 45, 11, 57, 14, 25, 33, 33, 45, 84, 44, 31, 36, 54, 61, 12, 8, 4, 13, 105, 76, 4, 55, 48, 20])

        elif num_classes == 100:
            self.class_counts = np.array([67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67])
        elif num_classes == 23:
            self.class_counts = np.array([8, 76, 9, 29, 60, 278, 40, 6, 13, 137, 105, 16, 502, 2000, 7, 48, 6, 5, 7, 57, 605, 2000, 145])
        else:
            self.class_counts = np.ones(num_classes)

        ## 根据样本数量初始化权重是不是不太合理，

        class_counts_tensor = torch.tensor(self.class_counts, dtype=torch.float32)
        initial_weights = self._initialize_smooth_weights(class_counts_tensor)

        self.init_weights = nn.Parameter(initial_weights, requires_grad=True)  # [num_classes]

        self.num_bins = num_bins

        ## 搞成5阶分箱
        bin_center = torch.linspace(0, 2, steps=self.num_bins + 1, dtype=torch.float32)
        self.register_buffer('bin_center', bin_center)

        ## top3,熵值，置信度
        self.projection = nn.Sequential(
            nn.Linear(5, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        self.fuse_weight = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, self.num_bins + 1),
        )


    def update_accuracy(self, vision_accuacy, text_accuacy):
        self.vision_accuacy = vision_accuacy
        self.text_accuacy = text_accuacy

    def _initialize_smooth_weights(self, class_counts):
        """
        根据样本数量初始化平滑的融合权重。
        样本数量越少，权重越大，且不进行总和归一化。
        """
        class_counts = class_counts.float()

        max_count = class_counts.max()
        min_count = class_counts.min()

        if max_count == min_count:
            # 如果所有类别计数都相同，可以给一个默认的非零权重，例如1.0
            return torch.ones_like(class_counts)

        # 归一化到 [0, 1] 范围，其中样本数少的映射到接近1，样本数多的映射到接近0
        # 使用1e-8避免除以零
        normalized_counts = 1.0 - (class_counts - min_count) / (max_count - min_count + 1e-8)

        beta = 2.0  # 可以根据需要调整，例如1.0为线性，2.0为平方
        smoothed_weights = torch.pow(normalized_counts, beta)

        # 不需要总和归一化，直接返回
        return smoothed_weights.float()

    def get_entropy(self, probs: torch.Tensor):
        # 1. 计算视觉分支的softmax概率
        vision_probs = F.softmax(probs, dim=-1)
        entropy = -torch.sum(vision_probs * torch.log(vision_probs + 1e-8), dim=-1)
        return entropy

    def get_confidence(self, logits):
        probs = F.softmax(logits, dim=-1)
        confidence, _ = torch.max(probs, dim=-1, keepdim=True)
        return confidence  # [B, 1]

    def get_confidence_embedding(self, logits, k=3):
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1, largest=True, sorted=True)
        batch_size = logits.size(0)
        embedding_weights = self.init_weights.to(logits.device).unsqueeze(0).expand(batch_size, -1)  # [B, num_classes]
        embedding_weights = embedding_weights.gather(1, topk_indices)  # [B, 3]
        embedding_weights = embedding_weights / (topk_probs + 1e-8) # [B, 3]

        return embedding_weights  # [B, 3]

    def get_topk_embedding(self, logits, k=3, type='confidence', reduce='sum'):
        if type == 'confidence':
            if reduce == 'sum':
                return self.get_confidence_embedding(logits, k=k).sum(dim=-1, keepdim=True)
            return self.get_confidence_embedding(logits, k=k)
        else:
            if reduce == 'sum':
                return self.get_entropy_embedding(logits, k=k).sum(dim=-1, keepdim=True)
            return self.get_entropy_embedding(logits, k=k)

    def get_entropy_embedding(self, logits, k=3):
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # [B]
        _, topk_indices = torch.topk(probs, k=k, dim=-1, largest=True, sorted=True)
        batch_size = logits.size(0)
        embedding_weights = self.init_weights.to(logits.device).unsqueeze(0).expand(batch_size, -1)  # [B, num_classes]
        embedding_weights = embedding_weights.gather(1, topk_indices)  # [B, 3]

        return embedding_weights * entropy.unsqueeze(-1)  # [B, 3]

    def get_accuacy_embedding(self, v_logits, t_logits, k=3):
        batch_size = v_logits.size(0)

        probs = F.softmax(v_logits, dim=-1)
        v_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # [B]
        _, topk_indices = torch.topk(probs, k=k, dim=-1, largest=True, sorted=True)
        v_embedding_weights = self.vision_accuacy.to(v_logits.device).unsqueeze(0).expand(batch_size, -1)  # [B, num_classes]
        v_embedding_weights = v_embedding_weights.gather(1, topk_indices)  # [B, 3]
        v_embedding_weights = v_embedding_weights * v_entropy.unsqueeze(-1)  # [B, 3]

        probs = F.softmax(t_logits, dim=-1)
        t_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # [B]
        _, topk_indices = torch.topk(probs, k=k, dim=-1, largest=True, sorted=True)
        t_embedding_weights = self.text_accuacy.to(t_logits.device).unsqueeze(0).expand(batch_size, -1)  # [B, num_classes]
        t_embedding_weights = t_embedding_weights.gather(1, topk_indices)  # [B, 3]
        t_embedding_weights = t_embedding_weights * t_entropy.unsqueeze(-1)  # [B, 3]

        return v_embedding_weights.sum(dim=-1, keepdim=True), t_embedding_weights.sum(dim=-1, keepdim=True)

    def forward(self, v_logits, t_logits):
        ## new表示可学习，explict表示不可学习
        batch_size = v_logits.size(0)

        probs = F.softmax(v_logits, dim=-1)
        v_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # [B]
        v_confidence, _ = torch.max(probs, dim=-1)  # [B]
        _, topk_indices = torch.topk(probs, k=3, dim=-1, largest=True, sorted=True)
        v_embedding_weights = self.init_weights.to(v_logits.device).unsqueeze(0).expand(batch_size, -1)  # [B, num_classes]
        v_embedding_weights = v_embedding_weights.gather(1, topk_indices)  # [B, 3]
        v_embedding = torch.cat([v_entropy.unsqueeze(-1), v_confidence.unsqueeze(-1), v_embedding_weights], dim=-1)  # [B, 5]
        v_f = self.projection(v_embedding)  # [B, 32]

        probs = F.softmax(t_logits, dim=-1)
        t_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # [B]
        t_confidence, _ = torch.max(probs, dim=-1)  # [B]
        _, topk_indices = torch.topk(probs, k=3, dim=-1, largest=True, sorted=True)
        t_embedding_weights = self.init_weights.to(t_logits.device).unsqueeze(0).expand(batch_size, -1)  # [B, num_classes]
        t_embedding_weights = t_embedding_weights.gather(1, topk_indices)  #
        t_embedding = torch.cat([t_entropy.unsqueeze(-1), t_confidence.unsqueeze(-1), t_embedding_weights], dim=-1)  # [B, 5]
        t_f = self.projection(t_embedding)  # [B, 32]

        gate = self.fuse_weight(torch.cat([v_f, t_f], dim=-1))  # [B, 5]

        ## 根据分箱结果加权
        bin_probs = F.softmax(gate, dim=-1)
        text_weights = torch.sum(bin_probs * self.bin_center, dim=-1, keepdim=True)
        fused_logits = text_weights * t_logits + v_logits

        return fused_logits, text_weights