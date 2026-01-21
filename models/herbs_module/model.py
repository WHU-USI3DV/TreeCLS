import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch.nn.functional as F
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Union
import copy
import sys
from models import open_clip
from data.dataset import WHU_CLASS_NAME as CLASS_NAME


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

def compute_cam_in_chunks(feat_dict, logits, target_layers, chunk_size=4):
    """
    显存优化版：分块计算 CAM
    """
    batch_size = logits.size(0)
    final_masks = {layer: [] for layer in target_layers}

    # 将 Batch 切片 (例如 16 -> 4, 4, 4, 4)
    for i in range(0, batch_size, chunk_size):
        end_idx = min(i + chunk_size, batch_size)

        # 1. 切片 Logits
        chunk_logits = logits[i:end_idx]

        # 2. 切片 Features (字典里的每个 Tensor 都要切)
        chunk_feats = {}
        for k, v in feat_dict.items():
            if v is not None:
                chunk_feats[k] = v[i:end_idx]

        # 3. 计算这一小块的 Mask
        # 注意：这里调用的是之前的 multi_layer_grad_cam
        # is_training=False 确保每一小块算完立刻释放图！
        chunk_masks = multi_layer_grad_cam(
            feature_dict=chunk_feats,
            logits=chunk_logits,
            target_layers=target_layers,
            is_training=False # 强制不保留图，算完即毁
        )

        # 4. 收集结果
        for layer_name, mask_tensor in chunk_masks.items():
            final_masks[layer_name].append(mask_tensor)

    # 5. 拼接回完整 Batch
    result = {}
    for layer_name, mask_list in final_masks.items():
        if len(mask_list) > 0:
            result[layer_name] = torch.cat(mask_list, dim=0)

    return result


def simple_grad_cam(features: torch.Tensor, logits: torch.Tensor, return_graph: bool):
    """
    即插即用的 Grad-CAM 计算函数。

    Args:
        features: [B, C, H, W] 或 [B, S, C]。必须是 requires_grad=True 的张量。
        logits:   [B, num_classes]。最终的分类输出。
        is_training: bool。当前是否处于训练模式 (决定是否保留计算图)。

    Returns:
        mask: [B, S] (归一化后的 heatmap)，已经 detach，无梯度。
              如果输入特征没有梯度信息，返回 None。
    """
    # 0. 快速检查：如果特征图不需要梯度，无法计算 CAM
    if not features.requires_grad:
        return None

    # 1. 强制开启梯度计算环境 (兼容 torch.no_grad 测试环境)
    with torch.enable_grad():
        # 确保 logits 也是在这个环境中计算出来的，如果 logits 已经断了，这里也无法回溯
        # 通常 logits 是由 features 算出来的，所以只要 features 有梯度，logits 就有 grad_fn

        # 2. 准备目标分数 (Target Score)
        # 取预测概率最大的类别分数总和
        probs = F.softmax(logits, dim=-1)
        target_ids = probs.argmax(dim=-1)
        target_score = logits.gather(1, target_ids.unsqueeze(-1)).sum()

        # 3. 计算梯度 (核心逻辑封装)
        # is_training=True  -> retain_graph=True  (为了后续 Loss Backward)
        # is_training=False -> retain_graph=False (算完即毁，防止显存爆炸)
        grads = torch.autograd.grad(
            outputs=target_score,
            inputs=features,
            grad_outputs=None,
            retain_graph=return_graph,
            create_graph=False,
            allow_unused=True
        )[0]

        if grads is None:
            return None

        # 4. 根据维度生成 CAM (自动适配 CNN/ViT)
        if len(features.shape) == 4: # [B, C, H, W]
            weights = torch.mean(grads, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * features, dim=1)
            cam = cam.view(cam.shape[0], -1) # Flatten -> [B, S]

        elif len(features.shape) == 3: # [B, S, C] or [B, C, S]
            # 自动判定 Channel 维
            if features.shape[-1] == grads.shape[-1]: # [B, S, C]
                weights = torch.mean(grads, dim=1, keepdim=True)
                cam = torch.sum(weights * features, dim=-1)
            else: # [B, C, S]
                weights = torch.mean(grads, dim=2, keepdim=True)
                cam = torch.sum(weights * features, dim=1)

        # 5. 后处理 (ReLU + Norm)
        cam = F.relu(cam)
        min_v = cam.min(dim=1, keepdim=True)[0]
        max_v = cam.max(dim=1, keepdim=True)[0]
        cam = (cam - min_v) / (max_v - min_v + 1e-8)

        ## 只保留前50%的cam，并二值化
        cam = (cam > cam.quantile(0.5, dim=1, keepdim=True)).float()

    # 6. 返回纯数值 (Detach)
    return cam.detach()

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
                size = inputs[name].size()
                if len(size) == 4:
                    in_size = size[1]
                elif len(size) == 3:
                    in_size = size[2]
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
        num_joints = total_num_selects // 64
        self.param_pool0 = nn.Linear(total_num_selects, num_joints)

        A = torch.eye(num_joints) / 100 + 1 / 100
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
        names = []

        for name in x:
            if "FPN1_" in name:
                continue
            if self.fpn_size is None:
                _tmp = getattr(self, "proj_"+name)(x[name])
            else:
                _tmp = x[name]
            hs.append(_tmp)
            names.append([name, _tmp.size()])

        hs = torch.cat(hs, dim=1).transpose(1, 2).contiguous() # B, S', C --> B, C, S
        # print(hs.size(), names)
        ## 剩下7个专家做自注意力
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

        _hs = hs.squeeze(-1)

        hs = self.dropout(hs)
        hs = hs.flatten(1)
        hs = self.classifier(hs)

        return hs, _hs

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
                # if in_dim != out_dim:
                m = nn.Conv1d(in_dim, out_dim, 1) # for spatial domain
                # else:
                #     m = nn.Identity()
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
            if "FPN1_" in name:
                continue
            x[name] = getattr(self, "Proj_"+name)(x[name])
            hs.append(name)

        x["FPN1_" + "layer4"] = x["layer4"]

        for i in range(len(hs)-1, 0, -1):
            x1_name = hs[i]
            x0_name = hs[i-1]
            x[x0_name] = self.upsample_add(x[x0_name],
                                           x[x1_name],
                                           x1_name)
            x["FPN1_" + x0_name] = x[x0_name]

        return x


class AdvancedSmoothFusion(nn.Module):
    """双参数平滑权重融合模块"""

    def __init__(self, init_temp=2.0, init_center=0.5, init_beta=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temp), requires_grad=True)
        self.center = nn.Parameter(torch.tensor(init_center), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(init_beta), requires_grad=True)

    def forward(self, vision_logits, text_logits, H_v):
        # 双参数平滑函数
        base_weight = torch.sigmoid(self.temperature * (H_v - self.center))
        text_weight = base_weight * (1 - self.beta) + self.beta
        text_weight = text_weight.view(-1, 1)
        vision_weight = 1.0 - text_weight

        return vision_weight * vision_logits + text_weight * text_logits, text_weight

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

        self.thresholds = {}
        for name in inputs:
            self.thresholds[name] = []

    # def select(self, logits, l_name):
    #     """
    #     logits: [B, S, num_classes]
    #     """
    #     probs = torch.softmax(logits, dim=-1)
    #     scores, _ = torch.max(probs, dim=-1)
    #     _, ids = torch.sort(scores, -1, descending=True)
    #     sn = self.num_select[l_name]
    #     s_ids = ids[:, :sn]
    #     not_s_ids = ids[:, sn:]
    #     return s_ids.unsqueeze(-1), not_s_ids.unsqueeze(-1)

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
        selections = {}
        for name in x:
            # print("[selector]", name, x[name].size())
            if "FPN1_" in name:
                continue
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                x[name] = x[name].view(B, C, H*W).permute(0, 2, 1).contiguous()
            C = x[name].size(-1)
            if self.fpn_size is None:
                logits[name] = getattr(self, "classifier_l_"+name)(x[name])
            probs = torch.softmax(logits[name], dim=-1)
            sum_probs = torch.softmax(logits[name].mean(1), dim=-1)
            ## 用来分类的
            selections[name] = []
            preds_1 = []
            preds_0 = []

            masks = []

            num_select = self.num_select[name]

            weights = []
            for bi in range(logits[name].size(0)):
                _, max_ids = torch.max(sum_probs[bi], dim=-1)
                confs, ranks = torch.sort(probs[bi, :, max_ids], descending=True)
                threshold = confs[num_select-1]
                weights.append(self.hard_threshold_mask(probs[bi, :, max_ids], threshold))
                sf = x[name][bi][ranks[:num_select]]
                nf = x[name][bi][ranks[num_select:]]  # calculate
                mask = self.get_mask(ranks, num_select, x[name].shape[1])
                masks.append(mask.unsqueeze(0))
                selections[name].append(sf) # [num_selected, C]
                preds_1.append(logits[name][bi][ranks[:num_select]])
                preds_0.append(logits[name][bi][ranks[num_select:]])

                if bi >= len(self.thresholds[name]):
                    self.thresholds[name].append(confs[num_select]) # for initialize
                else:
                    self.thresholds[name][bi] = confs[num_select]

            selections[name] = torch.stack(selections[name])
            preds_1 = torch.stack(preds_1)
            preds_0 = torch.stack(preds_0)

            logits["select_"+name] = preds_1
            logits["drop_"+name] = preds_0
            logits["weight_"+name.replace('layer','')] = torch.stack(weights).to(x[name].device)

            if not self.training:
                logits['mask_'+name] = torch.cat(masks, dim=0).to(x[name].device)
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
                # if in_dim != out_dim:
                m = nn.Conv1d(in_dim, out_dim, 1) # for spatial domain
                # else:
                #     m = nn.Identity()
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
            if "FPN1_" in name:
                continue
            x[name] = getattr(self, "Proj_"+name)(x[name])
            hs.append(name)

        x["FPN1_" + "layer4"] = x["layer4"]

        for i in range(len(hs)-1, 0, -1):
            x1_name = hs[i]
            x0_name = hs[i-1]
            x[x0_name] = self.upsample_add(x[x0_name],
                                           x[x1_name],
                                           x1_name)
            x["FPN1_" + x0_name] = x[x0_name]

        return x


class FPN_UP(nn.Module):

    def __init__(self,
                 inputs: dict,
                 fpn_size: int):
        super(FPN_UP, self).__init__()

        inp_names = [name for name in inputs]

        for i, node_name in enumerate(inputs):
            ### projection module
            m = nn.Sequential(
                nn.Linear(fpn_size, fpn_size),
                nn.ReLU(),
                nn.Linear(fpn_size, fpn_size),
            )
            self.add_module("Proj_"+node_name, m)

            ### upsample module
            if i != (len(inputs) - 1):
                assert len(inputs[node_name].size()) == 3 # B, S, C
                in_dim = inputs[node_name].size(1)
                out_dim = inputs[inp_names[i+1]].size(1)
                m = nn.Conv1d(in_dim, out_dim, 1) # for spatial domain
                self.add_module("Down_"+node_name, m)
                # print("Down_"+node_name, in_dim, out_dim)
                """
                Down_layer1 2304 576
                Down_layer2 576 144
                Down_layer3 144 144
                """

    def downsample_add(self, x0: torch.Tensor, x1: torch.Tensor, x0_name: str):
        """
        return Upsample(x1) + x1
        """
        # print("[downsample_add] Down_" + x0_name)
        x0 = getattr(self, "Down_" + x0_name)(x0)
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
            if "FPN1_" in name:
                continue
            x[name] = getattr(self, "Proj_"+name)(x[name])
            hs.append(name)

        # print(hs)
        for i in range(0, len(hs) - 1):
            x0_name = hs[i]
            x1_name = hs[i+1]
            # print(x0_name, x1_name)
            # print(x[x0_name].size(), x[x1_name].size())
            x[x1_name] = self.downsample_add(x[x0_name],
                                             x[x1_name],
                                             x0_name)
        return x


class DynamicRouter(nn.Module):
    def __init__(self, inputs, coarse_classes: int):
        """
        inputs: dictionary contain torch.Tensors, which comes from backbone
            [Tensor1(hidden feature1), Tensor2(hidden feature2)...]
            Please note that if len(features.size) equal to 3, the order of dimension must be [B,S,C],
            S mean the spatial domain, and if len(features.size) equal to 4, the order must be [B,C,H,W]
        coarse_classes: integer, number of coarse classes
        """
        super(DynamicRouter, self).__init__()
        input_dims = []
        self.coarse_classes = coarse_classes
        input_dims.append(inputs['layer1'].size(2))
        input_dims.append(inputs['layer2'].size(2))
        input_dims.append(inputs['layer3'].size(2))
        input_dims.append(inputs['layer4'].size(2))

        self.input_dim = input_dims

        sum_dim = sum(input_dims)

        self.coarse_linear = nn.Sequential(
            nn.Linear(sum_dim, sum_dim // 2),
            nn.ReLU(),
            nn.Linear(sum_dim // 2, coarse_classes)
        )

        # 树科专属参数（K个树科，每个有分类权重）
        self.tree_weights_0 = nn.Parameter(torch.randn(coarse_classes, input_dims[0]) * 0.01, requires_grad=True)
        self.tree_weights_1 = nn.Parameter(torch.randn(coarse_classes, input_dims[1]) * 0.01, requires_grad=True)
        self.tree_weights_2 = nn.Parameter(torch.randn(coarse_classes, input_dims[2]) * 0.01, requires_grad=True)
        self.tree_weights_3 = nn.Parameter(torch.randn(coarse_classes, input_dims[3]) * 0.01, requires_grad=True)

        self.tree_bias_0 = nn.Parameter(torch.zeros(coarse_classes, input_dims[0]), requires_grad=True)
        self.tree_bias_1 = nn.Parameter(torch.zeros(coarse_classes, input_dims[1]), requires_grad=True)
        self.tree_bias_2 = nn.Parameter(torch.zeros(coarse_classes, input_dims[2]), requires_grad=True)
        self.tree_bias_3 = nn.Parameter(torch.zeros(coarse_classes, input_dims[3]), requires_grad=True)

        # 温度参数（可学习或固定）
        self.tau = nn.Parameter(torch.ones(1) * 0.5, requires_grad=False)

    def forward(self, inputs):
        out = []
        for k in inputs.keys():
            out.append(inputs[k].mean(1))
        out = torch.cat(out, dim=1)
        coarse_logits = self.coarse_linear(out)
        # B, num_trees

        # 2. Gumbel-Softmax路由（核心！）
        if self.training:
            # 训练模式：软采样（梯度可穿透）
            weights = F.gumbel_softmax(
                logits=coarse_logits,
                tau=self.tau.item(),
                hard=False,  # 关键：训练时必须False
                dim=-1
            )
        else:
            # 推理模式：硬采样（one-hot）
            weights = F.gumbel_softmax(
                logits=coarse_logits,
                tau=0.1,  # 推理时用更低温度
                hard=True,
                dim=-1
            )

        # 3. 动态参数组合
        # 将树科权重与特征权重相乘：(B, K) x (K, C_out, C_in) -> (B, C_out, C_in)

        # 使用einsum正确组合权重: (B, num_classes, feat_dim)
        combined_weights0 = torch.einsum('bk,ki->bi', weights, self.tree_weights_0)
        combined_weights1 = torch.einsum('bk,ki->bi', weights, self.tree_weights_1)
        combined_weights2 = torch.einsum('bk,ki->bi', weights, self.tree_weights_2)
        combined_weights3 = torch.einsum('bk,ki->bi', weights, self.tree_weights_3)

        combined_bias0 = torch.einsum('bk,ki->bi', weights, self.tree_bias_0)
        combined_bias1 = torch.einsum('bk,ki->bi', weights, self.tree_bias_1)
        combined_bias2 = torch.einsum('bk,ki->bi', weights, self.tree_bias_2)
        combined_bias3 = torch.einsum('bk,ki->bi', weights, self.tree_bias_3)


        inputs['layer1'] = inputs['layer1'] * combined_weights0.unsqueeze(1) + combined_bias0.unsqueeze(1)
        inputs['layer2'] = inputs['layer2'] * combined_weights1.unsqueeze(1) + combined_bias1.unsqueeze(1)
        inputs['layer3'] = inputs['layer3'] * combined_weights2.unsqueeze(1) + combined_bias2.unsqueeze(1)
        inputs['layer4'] = inputs['layer4'] * combined_weights3.unsqueeze(1) + combined_bias3.unsqueeze(1)

        return coarse_logits, inputs

    def set_inference_mode(self, hard=True):
        """切换推理模式"""
        self.hard.fill_(hard)
        self.tau.fill_(0.1)  # 降低推理温度


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
                 use_cam: bool = False
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
        outs = self.backbone(rand_in)

        ### just original backbone
        if not use_fpn and (not use_selection and not use_combiner):
            for name in outs:
                fs_size = outs[name].size()
                if len(fs_size) == 3:
                    out_size = fs_size.size(-1)
                elif len(fs_size) == 4:
                    out_size = fs_size.size(1)
                else:
                    raise ValueError("The size of output dimension of previous must be 3 or 4.")
            self.classifier = nn.Linear(out_size, num_classes)

        ### = = = = = FPN = = = = =
        self.use_fpn = use_fpn
        if self.use_fpn:
            self.fpn_down = FPN(outs, fpn_size, proj_type, upsample_type)
            self.build_fpn_classifier_down(outs, fpn_size, num_classes)
            self.fpn_up = FPN_UP(outs, fpn_size)
            self.build_fpn_classifier_up(outs, fpn_size, num_classes)

        self.fpn_size = fpn_size

        ### = = = = = Selector = = = = =
        self.use_selection = use_selection
        if self.use_selection:
            w_fpn_size = self.fpn_size if self.use_fpn else None # if not using fpn, build classifier in weakly selector

            self.selector = WeaklySelector(outs, num_classes, num_selects, w_fpn_size)

        self.add_linear = add_linear
        self.add_feature = not add_linear

        self.only_loss = only_loss
        self.no_mask = no_mask

        self.add_loss = add_loss
        self.use_embedding = use_embedding

        self.use_cam = use_cam

        ### = = = = = Combiner = = = = =
        self.use_combiner = use_combiner
        if self.use_combiner:
            assert self.use_selection, "Please use selection module before combiner"
            if self.use_fpn:
                gcn_inputs, gcn_proj_size = None, None
            else:
                gcn_inputs, gcn_proj_size = outs, fpn_size # redundant, fix in future
            total_num_selects = sum([num_selects[name] for name in num_selects]) # sum
            c_fpn_size = self.fpn_size if self.use_fpn else None
            self.combiner = GCNCombiner(total_num_selects, num_classes, gcn_inputs, gcn_proj_size, c_fpn_size)

        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path)
            self.load_state_dict(ckpt['model'], strict=False)
            for param in self.parameters():
                param.requires_grad = False
            del ckpt
            self.frozen_backbone = True
        else:
            self.frozen_backbone = False

        if self.add_linear:
            ## 只训练以下参数
            self.prior_model = PriorModel(add_loss, no_mask, num_classes)
            self.text_pooling = nn.Linear(5, 1)
            self.classifier = nn.Linear(768, num_classes)
            if self.use_embedding:
                self.adn = LogitsFusion(num_classes)

    @property
    def beta(self):
        return torch.sigmoid(self.log_beta)

    def _reinit_plugins_weights(self):
        ## 使用默认初始化，初始化text_pooling和classifier
        for name, p in self.parameters():
            if name in ['prior_model', 'text_pooling', 'classifier']:
                continue
            else:
                p.requires_grad = False

        nn.init.xavier_uniform_(self.text_pooling.weight)
        nn.init.zeros_(self.text_pooling.bias)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)
        if self.use_embedding:
            self.adn._init_weights()

        ## 打印可训练参数总数
        total_params = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                param_count = p.numel()
                total_params += param_count
                print(f"  {name}: {param_count} parameters")
        print(f"[Re-init] Trainable parameters: {total_params}")

    def _init_embedding(self, num_classes):
        min_weight = 0.0
        max_weight = 9.0
        if num_classes == 200:
            num_samples_list = []
        elif num_classes == 102:
            num_samples_list = [66, 86, 55, 52, 67, 35, 89, 94, 24, 29, 34, 60, 43, 114, 64, 83, 59, 42, 12, 72, 47, 105, 132, 52, 23, 24, 26, 16, 32, 33, 66, 83, 42, 53, 8, 137, 34, 43, 21, 96, 12, 56, 63, 52, 12, 40, 24, 72, 68, 24, 32, 52, 15, 21, 24, 74, 14, 4, 64, 40, 20, 48, 15, 32, 10, 56, 42, 50, 16, 18, 3, 16, 4, 2, 21, 4, 48, 4, 45, 11, 57, 14, 25, 33, 33, 45, 84, 44, 31, 36, 54, 61, 12, 8, 4, 13, 105, 76, 4, 55, 48, 20]
        log_samples = np.log1p(num_samples_list)
        max_log = np.max(log_samples)
        min_log = np.min(log_samples)
        initial_weights = np.zeros(num_classes)
        for i, log_sample in enumerate(log_samples):
            # 归一化到[0,1]，然后反转（样本越少权重越高）
            normalized = 1.0 - (log_sample - min_log) / (max_log - min_log + 1e-8)
            initial_weights[i] = min_weight + (max_weight - min_weight) * normalized

        # 确保权重在[min_weight, max_weight]范围内
        initial_weights = np.clip(initial_weights, min_weight, max_weight)
        self.CLS_EMBEDDING.data.copy_(torch.tensor(initial_weights, dtype=torch.float32))

    def build_fpn_classifier_up(self, inputs: dict, fpn_size: int, num_classes: int):
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
            self.add_module("fpn_classifier_up_"+name, m)

    def build_fpn_classifier_down(self, inputs: dict, fpn_size: int, num_classes: int):
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
            self.add_module("fpn_classifier_down_" + name, m)

    def forward_backbone(self, x):
        return self.backbone(x)

    def fpn_predict_down(self, x: dict, logits: dict):
        """
        x: [B, C, H, W] or [B, S, C]
           [B, C, H, W] --> [B, H*W, C]
        """
        for name in x:
            if "FPN1_" not in name:
                continue
            ### predict on each features point
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                logit = x[name].view(B, C, H*W)
            elif len(x[name].size()) == 3:
                logit = x[name].transpose(1, 2).contiguous()
            model_name = name.replace("FPN1_", "")
            logits[name] = getattr(self, "fpn_classifier_down_" + model_name)(logit)
            logits[name] = logits[name].transpose(1, 2).contiguous() # transpose

    def fpn_predict_up(self, x: dict, logits: dict):
        """
        x: [B, C, H, W] or [B, S, C]
           [B, C, H, W] --> [B, H*W, C]
        """
        for name in x:
            if "FPN1_" in name:
                continue
            ### predict on each features point
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                logit = x[name].view(B, C, H*W)
            elif len(x[name].size()) == 3:
                logit = x[name].transpose(1, 2).contiguous()
            model_name = name.replace("FPN1_", "")
            logits[name] = getattr(self, "fpn_classifier_up_" + model_name)(logit)
            logits[name] = logits[name].transpose(1, 2).contiguous() # transpose

    def get_entropy(self, probs: torch.Tensor):
        """
        probs: [B, S, num_classes]
        """

        # 1. 计算视觉分支的softmax概率
        vision_probs = F.softmax(probs, dim=-1)

        # 2. 计算视觉分支的熵 (衡量不确定性)
        # 熵公式: H = -∑(p_i * log(p_i))
        entropy = -torch.sum(vision_probs * torch.log(vision_probs + 1e-8), dim=-1)

        return entropy

    def convert_normalize(self, tensor):
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

    def forward(self, x: torch.Tensor):
        logits = {}

        ori_img = self.convert_normalize(x)
        ori_img = F.interpolate(ori_img, size=(224, 224), mode='bicubic', align_corners=True)

        if self.use_cam:
            # === 模式 A: 开启 CAM (全程强制记录梯度) ===
            with torch.enable_grad():

                # 1. Backbone (处理冻结逻辑)
                if self.frozen_backbone:
                    # 冻结时：先计算数值(不存图)，再 detach 出来手动设为起点
                    with torch.no_grad():
                        feat_dict = self.forward_backbone(x)
                    feat_dict = {k: v.detach().requires_grad_(True) for k, v in feat_dict.items()}
                else:
                    # 训练时：正常计算，保留中间梯度
                    feat_dict = self.forward_backbone(x)
                    for v in feat_dict.values(): v.retain_grad()

                # 2. FPN (必须在 enable_grad 内部)
                if self.use_fpn:
                    feat_dict = self.fpn_down(feat_dict)
                    self.fpn_predict_down(feat_dict, logits)
                    feat_dict = self.fpn_up(feat_dict)
                    self.fpn_predict_up(feat_dict, logits)

                    # 确保 FPN 中间层也能被 CAM 可视化
                    for v in feat_dict.values():
                        if isinstance(v, torch.Tensor) and v.requires_grad:
                            v.retain_grad()

                # 3. Head (必须在 enable_grad 内部)
                if self.use_selection:
                    selects = self.selector(feat_dict, logits)

                comb_outs, _ = self.combiner(selects)

                prefix = "FPN1_" if self.use_fpn else ""
                # 构建需要计算的层名列表（这是 feature_dict 里的 key）
                target_layer_keys = [f'{prefix}{k}' for k in ['layer1', 'layer2', 'layer3', 'layer4']]

                generated_masks = multi_layer_grad_cam(
                    feature_dict=feat_dict,
                    logits=comb_outs,
                    target_layers=target_layer_keys,
                    is_training=self.training
                )
                # 将结果更新到 logits 中
                for full_key, mask in generated_masks.items():
                    # key处理: FPN1_layer1 -> weight_1
                    clean_key = full_key.replace("FPN1_", "").replace("layer", "weight_")
                    logits[clean_key] = mask
        else:
            # === 模式 B: 普通模式 (遵循 PyTorch 默认行为) ===
            feat_dict = self.forward_backbone(x)

            if self.use_fpn:
                feat_dict = self.fpn_down(feat_dict)
                self.fpn_predict_down(feat_dict, logits)
                feat_dict = self.fpn_up(feat_dict)
                self.fpn_predict_up(feat_dict, logits)

            if self.use_selection:
                selects = self.selector(feat_dict, logits)

            comb_outs, _ = self.combiner(selects)

            if self.add_linear:
                self.prior_model.eval()
                with torch.no_grad():
                    text_feature = self.prior_model(ori_img, logits)
                text_feature = text_feature.detach()
                text_feature = self.text_pooling(text_feature).squeeze(-1) # B, 768

                if self.add_loss and self.training:
                    if self.prior_model.text_features is None:
                        self.prior_model.get_text_embedding(text_feature.device)
                    clip_feature = F.normalize(text_feature, dim=-1)
                    logits[f'clip_loss'] = self.prior_model.logit_scale.exp() * clip_feature @ self.prior_model.text_features.t()

                out = self.classifier(text_feature)
                logits['text_outs'] = out
                logits['visual_outs'] = comb_outs

                ## for 消融logits融合
                if self.use_embedding:
                    fused_logits, text_weight = self.adn(comb_outs, out)
                else:
                    fused_logits = comb_outs + out
                    text_weight = torch.ones_like(fused_logits[:, :1])

                ## 测试一下不fuse,只有损失会怎么样
                if self.only_loss:
                    logits['comb_outs'] = comb_outs
                ## 只保留损失
                else:
                    logits['comb_outs'] = fused_logits

                logits['text_weight'] = text_weight.detach()
            else:
                logits['comb_outs'] = comb_outs

        if not self.training:
            # 遍历并创建一个新的字典，或者原地修改
            # 必须处理 logits 中的每一个 value，因为 evaluate 代码里访问了 outs['layer1'] 等
            for k, v in logits.items():
                if isinstance(v, torch.Tensor):
                    logits[k] = v.detach()

        return logits

class PriorModel(nn.Module):
    def __init__(self, add_loss=True, no_mask=False, num_classes=102):
        super(PriorModel, self).__init__()
        self.add_loss = add_loss
        self.no_mask = no_mask

        if num_classes == 100:
            ## 使用openclip的VIT-L模型
            self.model, _, self.tokenizer = open_clip.create_model_and_transforms(
                'ViT-L-14',
                pretrained='datacomp_xl_s13b_b90k',  # 指定预训练权重来源
            )
        else:
            self.model, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
            self.tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')

        self.class_text = ['a photo of ' +name.replace('_', ' ') for name in CLASS_NAME]
        self.text_features = None

        ## 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False

        if self.add_loss:
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))
            self.logit_scale.requires_grad = True

            for n, p in self.named_parameters():
                if p.requires_grad:
                    print("[PriorModel] training parameter:", n)

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
                weight = weight.view(B, 1, 16*16).squeeze(1) # B, 256
                clip_feature = self.model.encode_mask_image(x, weight)[0]

                feature.append(clip_feature)

        feature = torch.stack(feature, dim=1) # B, 4, 768
        feature = feature.permute(0, 2, 1).contiguous() # B, 768, 4

        return feature

    @torch.no_grad()
    def optimized_forward(self, x, logits: dict):
        ## 102, 768
        B = x.shape[0]
        feature = []
        clip_feature = self.model.encode_image(x, normalize=True)[0]
        feature.append(clip_feature)
        if self.no_mask:
            feature = [clip_feature] * 5
        else:
            weights = []
            for i in range(4):
                name = f'weight_{i+1}'
                weight = logits[name]
                H, W = int(weight.shape[1]**0.5), int(weight.shape[1]**0.5)
                weight = weight.view(B, H, W).unsqueeze(1) # B, 1, H, W
                weight = F.interpolate(weight, size=(16, 16), mode='bilinear', align_corners=True)
                weight = weight.view(B, 1, 16*16).squeeze(1) # B, 256
                weights.append(weight)
            weights = torch.cat(weights, dim=0) # 4*B, 256
            repeated_x = torch.cat([x] * 4, dim=0) # 4*B, 3, H, W
            mask_feature = self.model.encode_mask_image(repeated_x, weights)[0]
            feature = feature + list(mask_feature.split(B, dim=0))
        feature = torch.stack(feature, dim=1) # B, 4, 768
        feature = feature.permute(0, 2, 1).contiguous() # B, 768, 4

        return feature


class AdaptiveLogitsFusion(nn.Module):
    def __init__(self, num_classes, init_alpha=5.0, init_beta=0.3, init_temperature=1.0, init_center=0.5, init_e_temperature=1.0):
        """
        Args:
            class_counts: 每个类别的训练样本数量
            init_alpha: 非线性程度控制参数的初始值
            init_beta: 基础权重偏移的初始值
            init_temperature: 置信度平滑参数的初始值
        """
        super(AdaptiveLogitsFusion, self).__init__()

        if num_classes == 200:
            self.class_counts = []
        elif num_classes == 102:
            self.class_counts = torch.tensor([66, 86, 55, 52, 67, 35, 89, 94, 24, 29, 34, 60, 43, 114, 64, 83, 59, 42, 12, 72, 47, 105, 132, 52, 23, 24, 26, 16, 32, 33, 66, 83, 42, 53, 8, 137, 34, 43, 21, 96, 12, 56, 63, 52, 12, 40, 24, 72, 68, 24, 32, 52, 15, 21, 24, 74, 14, 4, 64, 40, 20, 48, 15, 32, 10, 56, 42, 50, 16, 18, 3, 16, 4, 2, 21, 4, 48, 4, 45, 11, 57, 14, 25, 33, 33, 45, 84, 44, 31, 36, 54, 61, 12, 8, 4, 13, 105, 76, 4, 55, 48, 20])

        # 将超参数设为可学习参数，使用适当的变换确保数值稳定性
        self.log_alpha = nn.Parameter(torch.tensor(float(np.log(init_alpha))), requires_grad=True)
        self.log_beta = nn.Parameter(torch.tensor(float(np.log(init_beta / (1 - init_beta)))), requires_grad=True)  # logit变换
        self.log_temperature = nn.Parameter(torch.tensor(float(np.log(init_temperature))), requires_grad=True)

        self.center = nn.Parameter(torch.tensor(float(init_center)), requires_grad=True)
        self.log_e_temperature = nn.Parameter(torch.tensor(float(np.log(init_e_temperature))), requires_grad=True)

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    @property
    def beta(self):
        return torch.sigmoid(self.log_beta)

    @property
    def temperature(self):
        return torch.exp(self.log_temperature)

    @property
    def e_temperature(self):
        return torch.exp(self.log_e_temperature)

    def _compute_class_weights(self):
        """基于样本数量计算类别权重"""
        # 归一化样本数量到 [0, 1]
        if self.class_counts.device != self.alpha.device:
            self.class_counts = self.class_counts.to(self.alpha.device)

        min_count = self.class_counts.min()
        max_count = self.class_counts.max()

        normalized_counts = (self.class_counts - min_count) / (max_count - min_count + 1e-8)
        # 非线性变换：小样本类别获得更高权重
        # 平滑权重

        class_weights = self.beta + (1 - self.beta) * torch.exp(-self.alpha * normalized_counts)

        return class_weights * self.temperature

    def forward(self, visual_logits: torch.Tensor, text_logits: torch.Tensor, entropy: torch.Tensor):
        """
        Args:
            visual_logits: 视觉分支的logits，形状为 [B, num_classes]
            text_logits: 文本分支的logits，形状为 [B, num_classes]
        Returns:
            fused_logits: 融合后的logits，形状为 [B, num_classes]
            text_weight: 文本分支的权重，形状为 [B, 1]
        """
        ## 下一步把entropy考虑进来
        text_weight = self._compute_class_weights()

        ## 熵值越小越接近0.5，越大越接近1
        visual_weight = torch.sigmoid(self.e_temperature * (entropy - self.center))
        visual_weight = 0.5 * (1 - visual_weight) + 0.5 #[0.5 - 1]

        fused_logits = visual_weight.unsqueeze(1) * visual_logits + text_weight * text_logits

        return fused_logits, visual_weight, text_weight

    def get_hyperparameters(self):
        """返回当前超参数值"""
        return {
            'alpha': self.alpha.item(),
            'beta': self.beta.item(),
            'temperature': self.temperature.item()
        }


class LogitsFusion_Back(nn.Module):
    def __init__(self, num_classes=102, fpn_size=1536):
        super(LogitsFusion_Back, self).__init__()

        if num_classes == 200:
            self.class_counts = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30]

        elif num_classes == 102:
            self.class_counts = np.array([66, 86, 55, 52, 67, 35, 89, 94, 24, 29, 34, 60, 43, 114, 64, 83, 59, 42, 12, 72, 47, 105, 132, 52, 23, 24, 26, 16, 32, 33, 66, 83, 42, 53, 8, 137, 34, 43, 21, 96, 12, 56, 63, 52, 12, 40, 24, 72, 68, 24, 32, 52, 15, 21, 24, 74, 14, 4, 64, 40, 20, 48, 15, 32, 10, 56, 42, 50, 16, 18, 3, 16, 4, 2, 21, 4, 48, 4, 45, 11, 57, 14, 25, 33, 33, 45, 84, 44, 31, 36, 54, 61, 12, 8, 4, 13, 105, 76, 4, 55, 48, 20])

        elif num_classes == 100:
            self.class_counts = np.array([67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67])

        self.fpn_size = fpn_size

        ## 初始化生成embedding权重
        log_samples = np.log1p(self.class_counts)
        max_log = np.max(log_samples)
        min_log = np.min(log_samples)
        initial_weights = np.zeros(num_classes)
        for i, log_sample in enumerate(log_samples):
            # 归一化到[0,1]，然后反转（样本越少权重越高）
            normalized = 1.0 - (log_sample - min_log) / (max_log - min_log + 1e-8)
            initial_weights[i] = normalized * 10000

        # self.init_weights = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32), requires_grad=True)  # [num_classes]
        self.init_weights = torch.tensor(initial_weights, dtype=torch.float32) # [num_classes]

        ## 把entropy乘法乘上去，即视觉的entropy越大以及咱们最混淆的类别越小样本，输入值越高，输入值越高，经过网络越接近1，文本权重越大。

        # 此外，仅考虑前三的text_logits，其他的不考虑
        ## 使用conv_bias或者使用一个简单的单调增函数来拟合这个关系
        ## bias的权值0-1

        self.embedding_proj = nn.Sequential(
            nn.Linear(num_classes * 3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        self.prior_proj = nn.Sequential(
            nn.Linear(fpn_size + 768, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        self.weight_proj = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def get_entropy(self, probs: torch.Tensor):
        # 1. 计算视觉分支的softmax概率
        vision_probs = F.softmax(probs, dim=-1)
        entropy = -torch.sum(vision_probs * torch.log(vision_probs + 1e-8), dim=-1)
        return entropy

    def get_embedding_weights(self, logits):
        ## 不用熵值了，改成置信度
        prob, top3_indices = torch.topk(F.softmax(logits, dim=-1), k=3, dim=-1, largest=True, sorted=True)
        ## 根据top三类别得到embdding权重
        batch_size = logits.size(0)

        embedding_weights = self.init_weights.to(logits.device).unsqueeze(0).expand(batch_size, -1)  # [B, num_classes]
        embedding_weights = embedding_weights.gather(1, top3_indices)  # [B, 3]
        embedding_weights = embedding_weights / (prob + 1e-8)  # [B, num_classes]

        return embedding_weights # [B, 3]

    def get_confidence_embedding(self, logits):
        probs = F.softmax(logits, dim=-1)
        confidence, _ = torch.max(probs, dim=-1, keepdim=True)
        return confidence * self.init_weights.to(logits.device).unsqueeze(0)  # [B, num_classes]

    def forward(self, vision_logits, text_logits, v_prior, t_prior):
        # 双参数平滑函数
        ## 得到置信度
        v_embedding_weights = self.get_confidence_embedding(vision_logits)  # [B, num_classes]
        t_embedding_weights = self.get_confidence_embedding(text_logits)  # [B, num_classes]
        combined_weights = torch.cat([v_embedding_weights, t_embedding_weights, v_embedding_weights - t_embedding_weights], dim=-1)  # [B, num_classes * 3]
        e_f = self.embedding_proj(combined_weights)  # [B, 1]
        p_f = self.prior_proj(torch.cat([v_prior, t_prior], dim=-1))  # [B, 1]
        text_weight = self.weight_proj(e_f + p_f)  # [B, 1]
        fused_logits = text_weight * text_logits + (1 - text_weight) * vision_logits

        return fused_logits, text_weight


class LogitsFusion(nn.Module):
    def __init__(self, num_classes=102):
        super(LogitsFusion, self).__init__()

        if num_classes == 200:
            self.class_counts = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30]

        elif num_classes == 102:
            self.class_counts = np.array([66, 86, 55, 52, 67, 35, 89, 94, 24, 29, 34, 60, 43, 114, 64, 83, 59, 42, 12, 72, 47, 105, 132, 52, 23, 24, 26, 16, 32, 33, 66, 83, 42, 53, 8, 137, 34, 43, 21, 96, 12, 56, 63, 52, 12, 40, 24, 72, 68, 24, 32, 52, 15, 21, 24, 74, 14, 4, 64, 40, 20, 48, 15, 32, 10, 56, 42, 50, 16, 18, 3, 16, 4, 2, 21, 4, 48, 4, 45, 11, 57, 14, 25, 33, 33, 45, 84, 44, 31, 36, 54, 61, 12, 8, 4, 13, 105, 76, 4, 55, 48, 20])

        elif num_classes == 100:
            self.class_counts = np.array([67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67])

        ## 根据样本数量初始化权重是不是不太合理，

        class_counts_tensor = torch.tensor(self.class_counts, dtype=torch.float32)
        initial_weights = self._initialize_smooth_weights(class_counts_tensor)

        self.init_weights = nn.Parameter(initial_weights, requires_grad=True)  # [num_classes]

        self.num_bins = 16

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

    def forward_explict(self, vision_logits, text_logits):
        v_embedding, t_embedding = self.get_accuacy_embedding(vision_logits, text_logits, k=3)  # [B, 3]
        gate = self.select_gate(torch.cat([v_embedding, t_embedding, v_embedding - t_embedding], dim=-1))
        ## 根据分箱结果加权
        bin_probs = F.softmax(gate, dim=-1)
        text_weights = torch.sum(bin_probs * self.bin_center, dim=-1, keepdim=True)
        fused_logits = text_weights * text_logits + vision_logits

        return fused_logits, text_weights

    def forward_856(self, v_logits, t_logits):
        batch_size = v_logits.size(0)

        probs = F.softmax(v_logits, dim=-1)
        v_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # [B]
        v_confidence, _ = torch.max(probs, dim=-1)  # [B]
        _, topk_indices = torch.topk(probs, k=3, dim=-1, largest=True, sorted=True)
        v_embedding_weights = self.vision_accuacy.to(v_logits.device).unsqueeze(0).expand(batch_size, -1)  # [B, num_classes]
        v_embedding_weights = v_embedding_weights.gather(1, topk_indices)  # [B, 3]
        v_embedding = torch.cat([v_entropy.unsqueeze(-1), v_confidence.unsqueeze(-1), v_embedding_weights], dim=-1)  # [B, 5]
        v_f = self.projection(v_embedding)  # [B, 32]

        probs = F.softmax(t_logits, dim=-1)
        t_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # [B]
        t_confidence, _ = torch.max(probs, dim=-1)  # [B]
        _, topk_indices = torch.topk(probs, k=3, dim=-1, largest=True, sorted=True)
        t_embedding_weights = self.text_accuacy.to(t_logits.device).unsqueeze(0).expand(batch_size, -1)  # [B, num_classes]
        t_embedding_weights = t_embedding_weights.gather(1, topk_indices)  #
        t_embedding = torch.cat([t_entropy.unsqueeze(-1), t_confidence.unsqueeze(-1), t_embedding_weights], dim=-1)  # [B, 5]
        t_f = self.projection(t_embedding)  # [B, 32]

        gate = self.fuse_weight(torch.cat([v_f, t_f], dim=-1))  # [B, 5]

        ## 根据分箱结果加权
        bin_probs = F.softmax(gate, dim=-1)
        text_weights = torch.sum(bin_probs * self.bin_center, dim=-1, keepdim=True)
        fused_logits = text_weights * t_logits + v_logits

        return fused_logits, text_weights

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


class UncertaintyGatedFusion(nn.Module):
    def __init__(self, num_classes=102):
        super(UncertaintyGatedFusion, self).__init__()

        if num_classes == 200:
            self.prior_counts = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30]

        elif num_classes == 102:
            self.prior_counts = np.array([66, 86, 55, 52, 67, 35, 89, 94, 24, 29, 34, 60, 43, 114, 64, 83, 59, 42, 12, 72, 47, 105, 132, 52, 23, 24, 26, 16, 32, 33, 66, 83, 42, 53, 8, 137, 34, 43, 21, 96, 12, 56, 63, 52, 12, 40, 24, 72, 68, 24, 32, 52, 15, 21, 24, 74, 14, 4, 64, 40, 20, 48, 15, 32, 10, 56, 42, 50, 16, 18, 3, 16, 4, 2, 21, 4, 48, 4, 45, 11, 57, 14, 25, 33, 33, 45, 84, 44, 31, 36, 54, 61, 12, 8, 4, 13, 105, 76, 4, 55, 48, 20])

        elif num_classes == 100:
            self.prior_counts = np.array([67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67, 67, 66, 67])

        # 2. 温度系数：学习缩放 Other Model 的 Logits，使其与 Baseline 分布匹配
        self.log_temperature = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_bias_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        # 3. 长尾 Logit 调整 (Logit Adjustment)
        # self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        freq = self.prior_counts / self.prior_counts.sum()
        adjustment = -np.log(freq + 1e-8)
        adjustment = (adjustment - adjustment.mean()) / (adjustment.std() + 1e-8)
        self.register_buffer('logit_bias', torch.tensor(adjustment, dtype=torch.float32))

    def _init_weights(self):
        self.log_temperature.data.fill_(0.0)
        self.log_bias_scale.data.fill_(0.0)

    def get_entropy(self, logits):
        """计算预测的熵，衡量不确定性"""
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1, keepdim=True)
        return entropy

    def get_stats(self, logits):
        """同时返回熵(Entropy)和最大置信度(Confidence)"""
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1, keepdim=True)
        confidence, _ = torch.max(probs, dim=-1, keepdim=True)
        return entropy, confidence

    def forward(self, logits_baseline, logits_other):

        temperature = torch.exp(self.log_temperature)
        logits_other_sharpened = logits_other / temperature

        fused_logits = logits_baseline + logits_other_sharpened + self.logit_bias.unsqueeze(0) * torch.exp(self.log_bias_scale)

        return fused_logits, temperature.unsqueeze(0)

    def forward_853(self, logits_baseline, logits_other):

        temperature = torch.clamp(self.temperature, min=0.0)

        fused_logits = logits_baseline + logits_other * temperature + self.logit_bias.unsqueeze(0) * torch.exp(self.log_bias_scale)

        return fused_logits, temperature.unsqueeze(0)

    def forward_back(self, logits_baseline, logits_other):

        logits_other = logits_other + self.logit_bias.unsqueeze(0) * torch.exp(self.log_bias_scale)
        temperature = torch.exp(self.log_temperature)

        fused_logits = logits_baseline + logits_other / temperature


        return fused_logits, temperature.unsqueeze(0)

class PlugAndPlayGradCAM(nn.Module):
    def __init__(self):
        super().__init__()
        # 不再需要初始化 target_layers，完全无状态

    def forward(self, features_map: dict, logits: torch.Tensor):
        """
        Args:
            features_map: 字典 { 'layer1': feat1, 'layer2': feat2, ... }
                          这里面的每一个 feat 都会被用来计算 mask。
                          Value 形状支持 [B, C, H, W] 或 [B, S, C] / [B, C, S]
            logits: 最终的视觉分类预测 [B, num_classes]
        Returns:
            masks: 字典 { 'grad_cam_mask_layer1': [B, S], ... }
        """
        masks = {}

        # 1. 准备反向传播的目标分数
        # 计算预测概率最高的类别的 Logits 之和
        probs = F.softmax(logits, dim=-1)
        target_ids = probs.argmax(dim=-1)
        one_hot = F.one_hot(target_ids, num_classes=logits.shape[-1]).float().to(logits.device)
        target_score = (logits * one_hot).sum()

        # 2. 直接遍历输入的字典
        for layer_name, feat in features_map.items():

            # --- 安全检查 ---
            # 如果特征没有梯度信息（未参与计算或被 detach），无法计算 CAM
            if not feat.requires_grad:
                continue

            # 3. 计算梯度
            # retain_graph=True: 必须保留计算图，因为后续还要对 logits 做真正的 loss backward
            # create_graph=False: Mask 生成过程本身不需要梯度
            grads = torch.autograd.grad(
                outputs=target_score,
                inputs=feat,
                grad_outputs=None,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )[0]

            if grads is None:
                continue

            # 4. 根据维度计算 CAM
            # [B, C, H, W] -> CNN / ResNet
            if len(feat.shape) == 4:
                # Global Average Pooling on gradients (dim 2,3)
                weights = torch.mean(grads, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * feat, dim=1)
                cam = cam.view(cam.shape[0], -1) # Flatten -> [B, S]

            # [B, S, C] or [B, C, S] -> Transformer / Swin
            elif len(feat.shape) == 3:
                # 自动判断 Channel 维：通常 Channel 维和 Gradient 的 Channel 维是一致的
                if feat.shape[-1] == grads.shape[-1]: # [B, S, C]
                    weights = torch.mean(grads, dim=1, keepdim=True) # [B, 1, C]
                    cam = torch.sum(weights * feat, dim=-1) # [B, S]
                else: # [B, C, S]
                    weights = torch.mean(grads, dim=2, keepdim=True) # [B, C, 1]
                    cam = torch.sum(weights * feat, dim=1) # [B, S]

            # 5. ReLU + Normalize
            cam = F.relu(cam)
            # 避免除以 0
            min_val = cam.min(dim=1, keepdim=True)[0]
            max_val = cam.max(dim=1, keepdim=True)[0]
            cam = (cam - min_val) / (max_val - min_val + 1e-8)

            ## 只保留前50%的cam，并二值化
            cam = (cam > cam.quantile(0.5, dim=1, keepdim=True)).float()

            # 存入结果，自动加上前缀
            masks[f'weight_{layer_name}'] = cam.detach()

        return masks