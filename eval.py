import torch
import warnings
torch.autograd.set_detect_anomaly(True)
warnings.simplefilter("ignore")
import torchvision
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import argparse
import timm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from data.dataset import WHUDataset, ImageDataset, AircraftDataset, RSTreeDataset
from tqdm import tqdm
from typing import Union

from utils.config_utils import load_yaml
from utils.evaluate import evaluate_tree_species, analyze_prediction_entropy
from utils.vis_utils import ImgLoader, get_cdict

global module_id_mapper
global features
global grads

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Union, Optional

from scipy.interpolate import make_interp_spline
import matplotlib.patches as mpatches

def visualize_logits_distribution(text_logits, vis_logits, pred_logits, text_weight, caption, gt, save_path="fusion_vis.png"):
    """
    可视化 Baseline (Visual) 和 Expert (Text) 的 Logits 分布（原始曲线）。

    特点:
    - 纯线条风格 (No Fill)，曲线基于原始 Logits (Softmax后)
    - X轴范围固定到1-101
    - 标题显示 Base/Expert/Fused 的预测类别及 GT
    - 术语更新为 Baseline / Expert
    """

    # --- 1. 数据转换 (Softmax后，不归一化) ---
    def to_numpy(x):
        return x.detach().cpu().numpy().flatten() if isinstance(x, torch.Tensor) else np.array(x).flatten()

    # 对所有 logits 应用 softmax 转换为概率分布
    v_prob = to_numpy(torch.softmax(vis_logits, dim=-1))
    t_prob = to_numpy(torch.softmax(text_logits, dim=-1))
    p_prob = to_numpy(torch.softmax(pred_logits * 2, dim=-1))

    # 提取权重
    w = text_weight.item() if isinstance(text_weight, torch.Tensor) else float(text_weight)

    # 计算预测结果 (0-indexed)
    base_pred_idx = np.argmax(v_prob)
    expert_pred_idx = np.argmax(t_prob)
    fused_pred_idx = np.argmax(p_prob)
    gt_idx = gt # 真实类别索引

    # 构建 Title 字符串 (使用 0-indexed)
    title_stats = f"Baseline Pred: {base_pred_idx}, Expert Pred: {expert_pred_idx}, Fuse Weight: {w:.2f}, Fused Pred: {fused_pred_idx}"
    full_title = f"{title_stats}\n{caption}" # 将描述作为副标题

    # 坐标轴数据
    num_classes = len(v_prob)
    x_indices = np.arange(num_classes) # 0 to num_classes-1

    # --- 2. 绘图设置 (SCI Style) ---
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 12,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # 颜色定义
    c_base = '#1f77b4'     # 蓝色 (Baseline)
    c_expert = '#ff7f0e'   # 橙色 (Expert)
    c_fused = '#d62728'    # 红色 (Fused)
    c_gt = '#2ca02c'       # 绿色 (GT)

    # --- 3. 绘制线条 (原始概率) ---

    # A. Baseline Probability (Visual) - 实线
    ax.plot(x_indices, v_prob, color=c_base, linestyle='-', linewidth=1.5, alpha=0.8, label='Baseline Probability')

    # B. Expert Probability (Text) - 虚线
    ax.plot(x_indices, t_prob, color=c_expert, linestyle='--', linewidth=1.5, alpha=0.9, label='Expert Probability')

    # C. Fused Probability - 实线，较粗
    ax.plot(x_indices, p_prob, color=c_fused, linestyle='-', linewidth=2.0, label='Fused Probability')

    # --- 4. 标注 GT 与 关键点 ---

    # GT 竖线 (虚线)
    ax.axvline(x=gt_idx, color=c_gt, linestyle='--', linewidth=1.5, alpha=0.8)
    # GT 顶部文字 (使用 0-indexed)
    ax.text(gt_idx, 1.05, f'GT={gt_idx}', color=c_gt, ha='center', va='bottom', fontweight='bold')

    # --- 5. 标题设置 & 布局 ---

    ax.set_title(full_title, fontsize=12, pad=15, linespacing=1.5)
    ax.set_xlabel('Class Index', fontweight='bold')
    ax.set_ylabel('Probability', fontweight='bold') # Y轴单位改为 Probability
    ax.set_ylim(-0.05, 1.15) # Y轴范围

    # 设置 X 轴范围为 0-100，并确保刻度清晰
    # 根据你的需求，X轴应该显示到101，包含100类。
    # 我们设置 xlim 为 -0.5 到 100.5, 这样 0 和 100 都能完整显示
    ax.set_xlim(-0.5, num_classes + 1.5)

    # 设置 X 轴刻度，如果类别太多，可能需要根据情况采样
    if num_classes <= 50:
        ax.set_xticks(np.arange(0, num_classes, 10)) # 每10个刻度显示一个
    else:
        # 如果类别多，比如100类，可以每20个显示一个，或根据焦点自动调整
        ax.set_xticks(np.arange(0, num_classes, 20))
        # 也可以考虑自动聚焦，但这里先按固定间隔显示

    # 图例
    ax.legend(loc='upper right', frameon=False, fontsize=10)

    # 去边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 网格
    ax.grid(axis='y', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


def save_patch_image(out, ori_img, args, save_path):
    for i in range(4):
        mask = out[f'mask_layer{i+1}']
        H = W = int(math.sqrt(mask.size(1)))
        mask = mask.view(H, W).cpu().numpy()
        mask = cv2.resize(mask, (args.data_size, args.data_size))
        masked_image = apply_transparent_mask(ori_img, mask, color=(0, 0, 255), alpha=0.4)
        cv2.imwrite(save_path + f"mask_{i+1}.png", masked_image)
        del mask, masked_image

def apply_transparent_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),  # BGR格式
    alpha: float = 0.4
) -> np.ndarray:
    """
    给图像应用透明掩膜

    参数:
    image: 原始图像 (H, W, 3), RGB或BGR格式
    mask: 二值掩膜 (H, W), 值为0或1 (也可以是0或255)
    color: 掩膜颜色 (B, G, R)
    alpha: 透明度系数 (0-1)

    返回:
    应用掩膜后的图像 (BGR格式)
    """
    # 确保输入图像为BGR格式（OpenCV标准）
    if len(image.shape) == 3 and image.shape[2] == 3:
        if np.all(image[:, :, 0] == image[:, :, 2]):  # 检测是否为RGB
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = image.copy()
    else:
        raise ValueError("图像必须是HWC格式的3通道图像")

    # 确保mask是二值的 (0和1)
    mask_binary = mask.copy()
    if mask_binary.max() > 1:
        mask_binary = (mask_binary > 0).astype(np.uint8)

    # 创建覆盖层
    overlay = img_bgr.copy()

    # 在覆盖层上应用颜色到mask区域
    overlay[:, :, 0] = np.where(mask_binary > 0, color[0], overlay[:, :, 0])
    overlay[:, :, 1] = np.where(mask_binary > 0, color[1], overlay[:, :, 1])
    overlay[:, :, 2] = np.where(mask_binary > 0, color[2], overlay[:, :, 2])

    # 混合图像
    result = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)

    return result

def count_numbers(data):
    count_dict = {}
    for sublist in data:
        for num in sublist:
            if num in count_dict:
                count_dict[num] += 1
            else:
                count_dict[num] = 1
    return count_dict

def plot_number_counts(count_dict, path):
    # 提取数字和对应的计数
    numbers = list(count_dict.keys())
    counts = list(count_dict.values())

    # 创建柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(numbers, counts, color='skyblue')

    # 在每个柱子上方添加数值标签
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 str(count), ha='center', va='bottom')

    # 设置图表标题和标签
    plt.title('Number Occurrence Count', fontsize=16)
    plt.xlabel('Numbers', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    # 美化图表
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(numbers)

    # 显示图表
    plt.tight_layout()
    plt.savefig(path)

def forward_hook(module: nn.Module, inp_hs, out_hs):
    global features, module_id_mapper
    layer_id = len(features) + 1
    module_id_mapper[module] = layer_id
    features[layer_id] = {}
    features[layer_id]["in"] = inp_hs
    features[layer_id]["out"] = out_hs
    # print('forward_hook, layer_id:{}, hs_size:{}'.format(layer_id, out_hs.size()))

def backward_hook(module: nn.Module, inp_grad, out_grad):
    global grads, module_id_mapper
    layer_id = module_id_mapper[module]
    grads[layer_id] = {}
    grads[layer_id]["in"] = inp_grad
    grads[layer_id]["out"] = out_grad
    # print('backward_hook, layer_id:{}, hs_size:{}'.format(layer_id, out_grad[0].size()))


def build_model(model_name:str,
                pretrained_path: str,
                img_size: int,
                fpn_size: int,
                num_classes: int,
                num_selects: dict,
                use_fpn: bool = True,
                use_selection: bool = True,
                use_combiner: bool = True,
                comb_proj_size: int = None,
                coarse_classes: Union[int, None] = None):

    from models.builder import MODEL_GETTER
    model = MODEL_GETTER[args.model_name](
        use_fpn = args.use_fpn,
        img_size = args.data_size,
        fpn_size = args.fpn_size,
        use_selection = args.use_selection,
        num_classes = args.num_classes,
        num_selects = args.num_selects,
        use_combiner = args.use_combiner,
        coarse_classes = coarse_classes,
        add_linear = getattr(args, "add_linear", False),
        feature_type = getattr(args, 'feature_type', None),
        add_loss = getattr(args, 'add_loss', False),
        only_loss = getattr(args, 'only_loss', False),
        no_mask = getattr(args, 'no_mask', False),
        use_embedding = getattr(args, 'use_embedding', False),
        pretrained_path = getattr(args, 'pretrained', None),
        use_cam = getattr(args, 'use_cam', False),
        num_bins=getattr(args, 'num_bins', 8),
        fuse_type=getattr(args, 'fuse_type', 0),
    ) # about return_nodes, we use our default setting

    if pretrained_path != "":
        ckpt = torch.load(pretrained_path)
        model.load_state_dict(ckpt['model'])

    model.eval()

    ### hook original layer1~4
    if getattr(model, 'layers', None) is not None:
        model.backbone.layers[0].register_forward_hook(forward_hook)
        model.backbone.layers[0].register_full_backward_hook(backward_hook)
        model.backbone.layers[1].register_forward_hook(forward_hook)
        model.backbone.layers[1].register_full_backward_hook(backward_hook)
        model.backbone.layers[2].register_forward_hook(forward_hook)
        model.backbone.layers[2].register_full_backward_hook(backward_hook)
        model.backbone.layers[3].register_forward_hook(forward_hook)
        model.backbone.layers[3].register_full_backward_hook(backward_hook)

    if getattr(model, 'fpn', None) is not None:
        ### hook original FPN layer1~4
        model.fpn.Proj_layer1.register_forward_hook(forward_hook)
        model.fpn.Proj_layer1.register_full_backward_hook(backward_hook)
        model.fpn.Proj_layer2.register_forward_hook(forward_hook)
        model.fpn.Proj_layer2.register_full_backward_hook(backward_hook)
        model.fpn.Proj_layer3.register_forward_hook(forward_hook)
        model.fpn.Proj_layer3.register_full_backward_hook(backward_hook)
        model.fpn.Proj_layer4.register_forward_hook(forward_hook)
        model.fpn.Proj_layer4.register_full_backward_hook(backward_hook)

    return model

def cal_backward(args, out, label, sum_type: str = "softmax"):
    assert sum_type in ["none", "softmax"]

    if getattr(args, "use_ori_model", True):
        if args.use_fpn:
            target_layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'FPN1_layer1', 'FPN1_layer2', 'FPN1_layer3', 'FPN1_layer4', 'comb_outs']
        else:
            target_layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'comb_outs']
    else:
        target_layer_names = ['comb_outs']

    sum_out = None
    for name in target_layer_names:

        if name != "comb_outs":
            tmp_out = out[name].mean(1)
        else:
            tmp_out = out[name]

        if sum_type == "softmax":
            tmp_out = torch.softmax(tmp_out, dim=-1)

        if sum_out is None:
            sum_out = tmp_out
        else:
            sum_out = sum_out + tmp_out # note that use '+=' would cause inplace error

    with torch.no_grad():
        if args.use_label:
            print("use label as target class")
            pred_score = torch.softmax(sum_out, dim=-1)[0][label]
            backward_cls = label
        else:
            pred_score, pred_cls = torch.max(torch.softmax(sum_out, dim=-1), dim=-1)
            pred_score = pred_score[0]
            pred_cls = pred_cls[0]
            backward_cls = pred_cls

    # print(sum_out.size())
    # print("pred: {}, gt: {}, score:{}".format(backward_cls, label, pred_score))
    if is_vis:
        sum_out[0, backward_cls].backward()


    return backward_cls

@torch.no_grad()
def get_grad_cam_weights(grads):
    weights = {}
    for grad_name in grads:
        _grad = grads[grad_name]['out'][0][0]
        L, C = _grad.size()
        H = W = int(L ** 0.5)
        _grad = _grad.view(H, W, C).permute(2, 0, 1)
        C, H, W = _grad.size()
        weights[grad_name] = _grad.mean(1).mean(1)
        # print(weights[grad_name].max())

    return weights

@torch.no_grad()
def plot_grad_cam(features, weights):
    act_maps = {}
    for name in features:
        hs = features[name]['out'][0]
        L, C = hs.size()
        H = W = int(L ** 0.5)
        hs = hs.view(H, W, C).permute(2, 0, 1)
        C, H, W = hs.size()
        w = weights[name]
        w = w.view(-1, 1, 1).repeat(1, H, W)
        weighted_hs = F.relu(w * hs)
        a_map = weighted_hs
        a_map = a_map.sum(0)
        # a_map /= abs(a_map).max()
        act_maps[name] = a_map
    return act_maps

if __name__ == "__main__":
    global module_id_mapper, features, grads, is_vis
    module_id_mapper, features, grads, is_vis = {}, {}, {}, False

    """
    Please add
    pretrained_path to yaml file.
    """
    # ===== 0. get setting =====
    parser = argparse.ArgumentParser("Visualize SwinT Large")
    parser.add_argument("-pr", "--pretrained_root", type=str,
        help="contain {pretrained_root}/best.pt, {pretrained_root}/config.yaml")
    parser.add_argument("-usl", "--use_label", default=False, type=bool)
    parser.add_argument("-sum_t", "--sum_features_type", default="softmax", type=str)
    parser.add_argument("-v", "--visualize", default=False, type=bool)
    parser.add_argument("-ec", "--evaluate_coarse", default=False, type=bool)
    parser.add_argument("--b", default=-1, type=int, help="num_bins")
    parser.add_argument("--f", default=-1, type=int, help="fuse_type")
    parser.add_argument("--m", default=-1, type=int, help="module_type")

    args = parser.parse_args()

    is_vis = args.visualize

    load_yaml(args, args.pretrained_root + "/config.yaml")

    coarse_classes = None

    # ===== 1. build model =====
    pretrained_path = args.pretrained_root + "/backup/best.pt"

    if args.b != -1:
        setattr(args, 'num_bins', args.b)
    if args.f != -1:
        setattr(args, 'fuse_type', args.f)

    if args.m == 1:
        setattr(args, 'model_name', 'resnet50')
    elif args.m == 2:
        setattr(args, 'model_name', 'swin-t')
    elif args.m == 3:
        setattr(args, 'model_name', 'vit')
    elif args.m == 4:
        setattr(args, 'model_name', 'bioclip')

    model = build_model(pretrained_path = pretrained_path,
                        img_size = args.data_size,
                        fpn_size = args.fpn_size,
                        use_fpn = args.use_fpn,
                        num_classes = args.num_classes,
                        num_selects = args.num_selects,
                        model_name=args.model_name,
                        coarse_classes=coarse_classes,
                    )
    model = model.cuda()

    # ===== 2. load image =====

    dataset_name = getattr(args, 'dataset', 'WHU')
    if dataset_name == 'WHU':
        test_set = WHUDataset(split='test', args=args, return_index=True)
    elif dataset_name == 'CUB':
        test_set = ImageDataset(istrain=False, root=args.val_root, data_size=args.data_size, return_index=True)
    elif dataset_name == 'Aircraft':
        test_set = AircraftDataset(root=args.val_root, train=False, data_size=args.data_size, return_index=True)
    elif dataset_name == 'RSTree':
        test_set = RSTreeDataset(split='test', args=args, return_index=True)

    test_loader = torch.utils.data.DataLoader(test_set, num_workers=args.num_workers, shuffle=False, batch_size=1)
    os.makedirs(args.pretrained_root + "/vis_imgs/Right", exist_ok=True)
    os.makedirs(args.pretrained_root + "/vis_imgs/Error", exist_ok=True)
    os.makedirs(args.pretrained_root + "/vis_imgs/Analyse/Right", exist_ok=True)
    os.makedirs(args.pretrained_root + "/vis_imgs/Analyse/Error", exist_ok=True)

    GT = []
    Pred = []
    Logists = []

    # ===== 3. forward and backward =====
    select_experts = []
    for batch_id, batch in enumerate(tqdm(test_loader)):
        ids, datas, labels = batch

        module_id_mapper, features, grads = {}, {}, {}
        label = labels.item()
        datas, labels = datas.cuda(), labels.cuda()

        out = model(datas)
        Logists.append(out['comb_outs'][0].detach().cpu().numpy())

        _, pred = torch.max(torch.softmax(out['comb_outs'], dim=-1), dim=-1)
        GT.append(label)

        Pred.append(pred.item())

        if is_vis:
            _, _pred = torch.max(torch.softmax(out['visual_outs'], dim=-1), dim=-1)
            if _pred != label and pred == label:
                ## 把这种类型的保存下来
                visualize_logits_distribution(out['text_outs'], out['visual_outs'], out['comb_outs'], out['text_weight'], '', label, args.pretrained_root + "/vis_imgs/Analyse/Right/{}.png".format(ids[0]))
                ##

evaluate_tree_species(Pred, GT, args.pretrained_root, num_classes=args.num_classes)

