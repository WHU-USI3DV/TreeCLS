import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


whu_class_names = ['Acer_palmatum', 'Aesculus_chinensis', 'Ailanthus_altissima', 'Albizia_julibrissin', 'Alstonia_scholaris', 'Araucaria_cunninghamii', 'Archontophoenix_alexandrae', 'Bauhinia_blakeana', 'Bauhinia_purpurea', 'Betula_platyphylla', 'Bischofia_polycarpa', 'Bombax_ceiba', 'Broussonetia_papyrifera', 'Caryota_maxima', 'Catalpa_ovata', 'Cedrus_deodara', 'Ceiba_speciosa', 'Celtis_sinensis', 'Celtis_tetrandra', 'Cinnamomum_camphora', 'Cinnamomum_japonicum', 'Citrus_maxima', 'Cocos_nucifera', 'Delonix_regia', 'Dimocarpus_longan', 'Dracontomelon_duperreanum', 'Elaeocarpus_decipiens', 'Elaeocarpus_glabripetalus', 'Erythrina_variegata', 'Eucommia_ulmoides', 'Euonymus_maackii', 'Euphorbia_milii', 'Ficus_altissima', 'Ficus_benjamina', 'Ficus_concinna', 'Ficus_microcarpa', 'Ficus_virens', 'Firmiana_simplex', 'Fraxinus_chinensis', 'Ginkgo_biloba', 'Grevillea_robusta', 'Koelreuteria_paniculata', 'Lagerstroemia_indica', 'Ligustrum_lucidum', 'Ligustrum_quihoui', 'Liquidambar_formosana', 'Liriodendron_chinense', 'Livistona_chinensis', 'Magnolia_grandiflora', 'Mangifera_persiciforma', 'Melia_azedarach', 'Metasequoia_glyptostroboides', 'Michelia_chapensis', 'Morella_rubra', 'Morus_alba', 'Osmanthus_fragrans', 'Paulownia_tomentosa', 'Phoebe_zhennan', 'Photinia_serratifolia', 'Picea_asperata', 'Picea_koraiensis', 'Picea_meyeri', 'Pinus_elliottii', 'Pinus_tabuliformis', 'Pinus_thunbergii', 'Pittosporum_tobira', 'Platanus', 'Platycladus_orientalis', 'Populus_alba', 'Populus_canadensis', 'Populus_cathayana', 'Populus_davidiana', 'Populus_hopeiensis', 'Populus_nigra', 'Populus_simonii', 'Populus_tomentosa', 'Prunus_cerasifera', 'Prunus_mandshurica', 'Prunus_salicina', 'Pseudolarix_amabilis', 'Pterocarya_stenoptera', 'Quercus_robur', 'Rhododendron_simsii', 'Robinia_pseudoacacia', 'Roystonea_regia', 'Sabina_chinensis', 'Salix_babylonica', 'Salix_matsudana', 'Sapindus_mukorossi', 'Sterculia_lanceolata', 'Styphnolobium_japonicum', 'Syringa_reticulata_subsp_amurensis', 'Syringa_villosa', 'Tilia_amurensis', 'Tilia_mandshurica', 'Toona_sinensis', 'Trachycarpus_fortunei', 'Triadica_sebifera', 'Ulmus_densa', 'Ulmus_pumila', 'Yulania_denudata', 'Zelkova_serrata']

CUB_CLASS_NAME = ['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet', 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher', 'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard', 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk', 'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow', 'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler', 'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat']


def analyze_confusion_pairs(cm, cm_normalized, class_names, save_dir, top_k=3):
    """
    分析每个类别最容易被混淆成哪些类别
    参数:
        cm_normalized: 归一化混淆矩阵 (已去对角线或未去均可)
        class_names: 类别名称列表
        save_dir: 保存路径
        top_k: 每个类输出前 K 个最易混淆类别
    """
    num_classes = cm_normalized.shape[0]

    # 创建结果列表
    results = []

    for i in range(num_classes):
        # 获取第 i 行（真实为 i 类），排除对角线
        row = cm_normalized[i].copy()
        row[i] = 0  # 确保对角线为 0（即使之前没处理）

        # 找出前 K 个最大值的索引
        top_k_idx = np.argsort(row)[::-1][:top_k]
        top_k_values = row[top_k_idx]

        # 构建每一行的输出
        result_row = {
            'True_Class_ID': i,
            'True_Class_Name': class_names[i],
            'Total_Samples': int(cm.sum(axis=1)[i]) if hasattr(cm, 'sum') else 'N/A'
        }

        for k in range(top_k):
            pred_id = top_k_idx[k]
            error_rate = top_k_values[k]
            result_row[f'Most_Confused_Label_{k+1}_ID'] = pred_id
            result_row[f'Most_Confused_Label_{k+1}_Name'] = class_names[pred_id]
            result_row[f'Error_Rate_{k+1}'] = round(error_rate, 4)

        results.append(result_row)

    # 转为 DataFrame 并保存
    confusion_df = pd.DataFrame(results)
    confusion_df.to_csv(os.path.join(save_dir, "confusion_analysis_per_class.csv"), index=False)

    print(f"\n✅ 每个类别的混淆分析已保存至:")
    print(f"   {os.path.join(save_dir, 'confusion_analysis_per_class.csv')}")
    print(f"   每个类显示前 {top_k} 个最易混淆的预测类别")

    return confusion_df


def evaluate_tree_species(Pred, GT,log_dir, num_classes, class_names=None):
    """
    树种分类任务评估函数
    输入：
        GT: 真实标签列表 [N,]
        Pred: 预测标签列表 [N,]
        class_names: 可选，字符串列表，如 ['Pinus tabuliformis', 'Quercus mongolica', ...]
    输出：
        打印并保存：Top-1 准确率、Precision、Recall、F1（macro & weighted）
        保存：CSV 汇总 + 混淆矩阵 + 分类报告 + 错误分析 + 可视化图
    """
    # 转换为 numpy
    GT_np = np.array(GT)
    Pred_np = np.array(Pred)

    # 创建保存目录
    save_dir = os.path.join(log_dir, "vis_imgs")
    os.makedirs(save_dir, exist_ok=True)

    # === 1. 计算 Top-1 准确度 ===
    top1_acc = accuracy_score(GT_np, Pred_np)
    print(f"Top-1 Accuracy: {top1_acc:.4f}")

    # === 2. 分类报告（精确度、召回率、F1）===
    if class_names is None:
        if num_classes == 200:
            class_names = CUB_CLASS_NAME
        elif num_classes == 102:
            class_names = whu_class_names
        else:
            class_names = [f"Class_{i}" for i in range(num_classes)]

    report = classification_report(GT_np, Pred_np, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(save_dir, "classification_report.csv"))
    print("\nDetailed Classification Report:")
    print(classification_report(GT_np, Pred_np, target_names=class_names))

    # 提取 macro avg 和 weighted avg 指标
    metrics_summary = {
        "Metric": ["Top-1 Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)",
                   "Precision (Weighted)", "Recall (Weighted)", "F1-Score (Weighted)"],
        "Value": [
            round(top1_acc, 4),
            round(report['macro avg']['precision'], 4),
            round(report['macro avg']['recall'], 4),
            round(report['macro avg']['f1-score'], 4),
            round(report['weighted avg']['precision'], 4),
            round(report['weighted avg']['recall'], 4),
            round(report['weighted avg']['f1-score'], 4),
        ]
    }
    summary_df = pd.DataFrame(metrics_summary)
    summary_df.to_csv(os.path.join(save_dir, "evaluation_summary.csv"), index=False)
    print("\n=== Evaluation Summary ===")
    print(summary_df.to_string(index=False))

    # === 3. 混淆矩阵 ===
    cm = confusion_matrix(GT_np, Pred_np)
    np.save(os.path.join(save_dir, "confusion_matrix.npy"), cm)
    pd.DataFrame(cm).to_csv(os.path.join(save_dir, "confusion_matrix.csv"), index=False)

    # 高分辨率混淆矩阵图
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar_kws={'shrink': 0.8})
    plt.title('Confusion Matrix (Tree Species)', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix_highres.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 归一化混淆矩阵（错误率，对角线置零）
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)  # 防除零
    np.fill_diagonal(cm_normalized, 0)

    analyze_confusion_pairs(
        cm=cm,
        cm_normalized=cm_normalized,
        class_names=class_names,
        save_dir=save_dir,
        top_k=5  # 可改为 5 如果需要更多
    )

    plt.figure(figsize=(20, 16))
    sns.heatmap(cm_normalized, annot=False, cmap='Reds', cbar_kws={'shrink': 0.8, 'label': 'Error Rate'})
    plt.title('Normalized Confusion Matrix (Errors Only)', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix_normalized.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # === 4. 错误分析：Top-20 最易错类别 ===
    errors_per_class = cm.sum(axis=1) - np.diag(cm)
    error_df = pd.DataFrame({
        'Class': range(num_classes),
        'Class_Name': class_names,
        'Samples': cm.sum(axis=1),
        'Errors': errors_per_class
    })
    error_df['ErrorRate'] = error_df['Errors'] / (error_df['Samples'] + 1e-8)
    error_df = error_df.sort_values('Errors', ascending=False)
    error_df.to_csv(os.path.join(save_dir, "error_analysis.csv"), index=False)

    # Top-20 错误最多类别柱状图
    plt.figure(figsize=(12, 8))
    top_errors = error_df.head(20)
    plt.bar(range(len(top_errors)), top_errors['Errors'])
    plt.xticks(range(len(top_errors)), top_errors['Class_Name'], rotation=60, ha='right')
    plt.title('Top 20 Classes with Most Errors')
    plt.ylabel('Number of Errors')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "top_errors.png"), dpi=150)
    plt.close()

    # === 5. 每个类别的准确率分析 ===
    class_accuracy = {}
    for i in range(num_classes):
        class_mask = (GT_np == i)
        if np.sum(class_mask) > 0:
            class_accuracy[class_names[i]] = accuracy_score(GT_np[class_mask], Pred_np[class_mask])
        else:
            class_accuracy[class_names[i]] = 0.0

    accuracy_df = pd.DataFrame({
        'Class': list(class_accuracy.keys()),
        'Accuracy': list(class_accuracy.values())
    })
    accuracy_df = accuracy_df.sort_values('Accuracy')
    accuracy_df.to_csv(os.path.join(save_dir, "class_accuracy.csv"), index=False)

    # 准确率分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(accuracy_df['Accuracy'], bins=20, edgecolor='black', color='skyblue')
    plt.title('Distribution of Class-wise Accuracy')
    plt.xlabel('Accuracy')
    plt.ylabel('Number of Classes')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_distribution.png"), dpi=150)
    plt.close()

    # === 输出保存信息 ===
    print(f"\n✅ 所有评估结果已保存至: {save_dir}")
    print("生成的文件包括:")
    print("  - evaluation_summary.csv (关键指标汇总)")
    print("  - classification_report.csv")
    print("  - confusion_matrix.npy/.csv")
    print("  - confusion_matrix_highres.png / normalized.png")
    print("  - error_analysis.csv / top_errors.png")
    print("  - class_accuracy.csv / accuracy_distribution.png")

    return metrics_summary  # 可用于后续调用

def suppression(target: torch.Tensor, threshold: torch.Tensor, temperature: float = 2):
    """
    target size: [B, S, C]
    threshold: [B',]
    """
    B = target.size(0)
    target = torch.softmax(target / temperature, dim=-1)
    # target = 1 - target
    return target

@torch.no_grad()
def cal_train_metrics(args, msg: dict, outs: dict, labels: torch.Tensor, batch_size: int, thresholds: dict):
    """
    only present top-1 training accuracy
    """

    total_loss = 0.0

    if args.use_fpn:
        for i in range(1, 5):
            acc = top_k_corrects(outs["layer"+str(i)].mean(1), labels, tops=[1])["top-1"] / batch_size
            acc = round(acc * 100, 2)
            msg["train_acc/layer{}_acc".format(i)] = acc
            loss = F.cross_entropy(outs["layer"+str(i)].mean(1), labels)
            msg["train_loss/layer{}_loss".format(i)] = loss.item()
            total_loss += loss.item()

            gt_score_map = outs["layer"+str(i)]
            thres = torch.Tensor(thresholds["layer"+str(i)])
            gt_score_map = suppression(gt_score_map, thres)
            logit = F.log_softmax(outs["FPN1_layer" + str(i)] / args.temperature, dim=-1)
            loss_b0 = nn.KLDivLoss()(logit, gt_score_map)
            msg["train_loss/layer{}_FPN1_loss".format(i)] = loss_b0.item()


    if args.use_selection:
        for name in outs:
            if "select_" not in name:
                continue
            B, S, _ = outs[name].size()
            logit = outs[name].view(-1, args.num_classes)
            labels_0 = labels.unsqueeze(1).repeat(1, S).flatten(0)
            acc = top_k_corrects(logit, labels_0, tops=[1])["top-1"] / (B*S)
            acc = round(acc * 100, 2)
            msg["train_acc/{}_acc".format(name)] = acc
            labels_0 = torch.zeros([B * S, args.num_classes]) - 1
            labels_0 = labels_0.to(args.device)
            loss = F.mse_loss(F.tanh(logit), labels_0)
            msg["train_loss/{}_loss".format(name)] = loss.item()
            total_loss += loss.item()

        for name in outs:
            if "drop_" not in name:
                continue
            B, S, _ = outs[name].size()
            logit = outs[name].view(-1, args.num_classes)
            labels_1 = labels.unsqueeze(1).repeat(1, S).flatten(0)
            acc = top_k_corrects(logit, labels_1, tops=[1])["top-1"] / (B*S)
            acc = round(acc * 100, 2)
            msg["train_acc/{}_acc".format(name)] = acc
            loss = F.cross_entropy(logit, labels_1)
            msg["train_loss/{}_loss".format(name)] = loss.item()
            total_loss += loss.item()

    if args.use_combiner:
        acc = top_k_corrects(outs['comb_outs'], labels, tops=[1])["top-1"] / batch_size
        acc = round(acc * 100, 2)
        msg["train_acc/combiner_acc"] = acc
        loss = F.cross_entropy(outs['comb_outs'], labels)
        msg["train_loss/combiner_loss"] = loss.item()
        total_loss += loss.item()

    if "ori_out" in outs:
        acc = top_k_corrects(outs["ori_out"], labels, tops=[1])["top-1"] / batch_size
        acc = round(acc * 100, 2)
        msg["train_acc/ori_acc"] = acc
        loss = F.cross_entropy(outs["ori_out"], labels)
        msg["train_loss/ori_loss"] = loss.item()
        total_loss += loss.item()

    msg["train_loss/total_loss"] = total_loss



@torch.no_grad()
def top_k_corrects(preds: torch.Tensor, labels: torch.Tensor, tops: list = [1, 3, 5]):
    """
    preds: [B, C] (C is num_classes)
    labels: [B, ]
    """
    if preds.device != torch.device('cpu'):
        preds = preds.cpu()
    if labels.device != torch.device('cpu'):
        labels = labels.cpu()
    tmp_cor = 0
    corrects = {"top-"+str(x):0 for x in tops}
    sorted_preds = torch.sort(preds, dim=-1, descending=True)[1]
    for i in range(tops[-1]):
        tmp_cor += sorted_preds[:, i].eq(labels).sum().item()
        # records
        if "top-"+str(i+1) in corrects:
            corrects["top-"+str(i+1)] = tmp_cor
    return corrects


@torch.no_grad()
def _cal_evalute_metric(corrects: dict,
                        total_samples: dict,
                        logits: torch.Tensor,
                        labels: torch.Tensor,
                        this_name: str,
                        scores: Union[list, None] = None,
                        score_names: Union[list, None] = None):

    tmp_score = torch.softmax(logits, dim=-1)
    tmp_corrects = top_k_corrects(tmp_score, labels, tops=[1, 3]) # return top-1, top-3, top-5 accuracy

    ### each layer's top-1, top-3 accuracy
    for name in tmp_corrects:
        eval_name = this_name + "-" + name
        if eval_name not in corrects:
            corrects[eval_name] = 0
            total_samples[eval_name] = 0
        corrects[eval_name] += tmp_corrects[name]
        total_samples[eval_name] += labels.size(0)

    if scores is not None:
        scores.append(tmp_score)
    if score_names is not None:
        score_names.append(this_name)


@torch.no_grad()
def _average_top_k_result(corrects: dict, total_samples: dict, scores: list, labels: torch.Tensor,
    tops: list = [1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """
    scores is a list contain:
    [
        tensor1,
        tensor2,...
    ] tensor1 and tensor2 have same size [B, num_classes]
    """
    # initial
    for t in tops:
        eval_name = "highest-{}".format(t)
        if eval_name not in corrects:
            corrects[eval_name] = 0
            total_samples[eval_name] = 0
        total_samples[eval_name] += labels.size(0)

    if labels.device != torch.device('cpu'):
        labels = labels.cpu()

    batch_size = labels.size(0)
    scores_t = torch.cat([s.unsqueeze(1) for s in scores], dim=1) # B, 5, C

    if scores_t.device != torch.device('cpu'):
        scores_t = scores_t.cpu()

    max_scores = torch.max(scores_t, dim=-1)[0]
    # sorted_ids = torch.sort(max_scores, dim=-1, descending=True)[1] # this id represents different layers outputs, not samples

    for b in range(batch_size):
        tmp_logit = None
        ids = torch.sort(max_scores[b], dim=-1)[1] # S
        for i in range(tops[-1]):
            top_i_id = ids[i]
            if tmp_logit is None:
                tmp_logit = scores_t[b][top_i_id]
            else:
                tmp_logit += scores_t[b][top_i_id]
            # record results
            if i+1 in tops:
                if torch.max(tmp_logit, dim=-1)[1] == labels[b]:
                    eval_name = "highest-{}".format(i+1)
                    corrects[eval_name] += 1


def evaluate(args, model, test_loader):
    """
    [Notice: Costom Model]
    If you use costom model, please change fpn module return name (under
    if args.use_fpn: ...)
    [Evaluation Metrics]
    We calculate each layers accuracy, combiner accuracy and average-higest-1 ~
    average-higest-5 accuracy (average-higest-5 means average all predict scores
    as final predict)
    """

    model.eval()
    corrects = {}
    total_samples = {}

    total_batchs = len(test_loader) # just for log
    show_progress = [x/10 for x in range(11)] # just for log
    progress_i = 0

    with torch.no_grad():
        """ accumulate """
        for batch_id, batch in enumerate(test_loader):
            ids, datas, labels = batch

            score_names = []
            scores = []
            datas = datas.to(args.device)
            outs = model(datas)

            if args.use_fpn:
                for i in range(1, 5):
                    this_name = "layer" + str(i)
                    if this_name in outs:
                        _cal_evalute_metric(corrects, total_samples, outs[this_name].mean(1), labels, this_name, scores, score_names)

            if args.use_selection:

                for name in outs:
                    if "select_" not in name:
                        continue
                    this_name = name
                    S = outs[name].size(1)
                    logit = outs[name].view(-1, args.num_classes)
                    labels_1 = labels.unsqueeze(1).repeat(1, S).flatten(0)
                    _cal_evalute_metric(corrects, total_samples, logit, labels_1, this_name)

                for name in outs:
                    if "drop_" not in name:
                        continue
                    this_name = name
                    S = outs[name].size(1)
                    logit = outs[name].view(-1, args.num_classes)
                    labels_0 = labels.unsqueeze(1).repeat(1, S).flatten(0)
                    _cal_evalute_metric(corrects, total_samples, logit, labels_0, this_name)

            if args.use_combiner:
                this_name = "combiner"
                _cal_evalute_metric(corrects, total_samples, outs["comb_outs"], labels, this_name, scores, score_names)

            if "ori_out" in outs:
                this_name = "original"
                _cal_evalute_metric(corrects, total_samples, outs["ori_out"], labels, this_name)


            _average_top_k_result(corrects, total_samples, scores, labels, tops=list(range(1, len(scores)+1)))

            eval_progress = (batch_id + 1) / total_batchs

            if eval_progress > show_progress[progress_i]:
                print(".."+str(int(show_progress[progress_i]*100))+"%", end='', flush=True)
                progress_i += 1

        """ calculate accuracy """
        # total_samples = len(test_loader.dataset)

        best_top1 = 0.0
        best_top1_name = ""
        eval_acces = {}
        for name in corrects:
            acc = corrects[name] / total_samples[name]
            acc = round(100 * acc, 3)
            eval_acces[name] = acc
            ### only compare top-1 accuracy
            if "top-1" in name or "highest" in name:
                if acc >= best_top1:
                    best_top1 = acc
                    best_top1_name = name

    return best_top1, best_top1_name, eval_acces


def evaluate_cm(args, model, test_loader):
    """
    [Notice: Costom Model]
    If you use costom model, please change fpn module return name (under
    if args.use_fpn: ...)
    [Evaluation Metrics]
    We calculate each layers accuracy, combiner accuracy and average-higest-1 ~
    average-higest-5 accuracy (average-higest-5 means average all predict scores
    as final predict)
    """

    model.eval()
    corrects = {}
    total_samples = {}
    results = []

    with torch.no_grad():
        """ accumulate """
        for batch_id, batch in enumerate(test_loader):
            if args.coarse_map:
                ids, datas, labels, coarse_labels = batch
            else:
                ids, datas, labels = batch

            score_names = []
            scores = []
            datas = datas.to(args.device)
            outs = model(datas, coarse_labels)

            # if args.use_fpn and (0 < args.highest < 5):
            #     this_name = "layer" + str(args.highest)
            #     _cal_evalute_metric(corrects, total_samples, outs[this_name].mean(1), labels, this_name, scores, score_names)

            if args.use_combiner:
                this_name = "combiner"
                _cal_evalute_metric(corrects, total_samples, outs["comb_outs"], labels, this_name, scores, score_names)

            # _average_top_k_result(corrects, total_samples, scores, labels)

            for i in range(scores[0].shape[0]):
                results.append([test_loader.dataset.data_infos[ids[i].item()]['path'], int(labels[i].item()),
                                int(scores[0][i].argmax().item()),
                                scores[0][i][scores[0][i].argmax().item()].item()])  # 图片路径，标签，预测标签，得分

        """ wirte xlsx"""
        writer = pd.ExcelWriter(args.save_dir + 'infer_result.xlsx')
        df = pd.DataFrame(results, columns=["id", "original_label", "predict_label", "goal"])
        df.to_excel(writer, index=False, sheet_name="Sheet1")
        writer.save()
        writer.close()

        """ calculate accuracy """

        best_top1 = 0.0
        best_top1_name = ""
        eval_acces = {}
        for name in corrects:
            acc = corrects[name] / total_samples[name]
            acc = round(100 * acc, 3)
            eval_acces[name] = acc
            ### only compare top-1 accuracy
            if "top-1" in name or "highest" in name:
                if acc > best_top1:
                    best_top1 = acc
                    best_top1_name = name

        """ wirte xlsx"""
        results_mat = np.mat(results)
        y_actual = results_mat[:, 1].transpose().tolist()[0]
        y_actual = list(map(int, y_actual))
        y_predict = results_mat[:, 2].transpose().tolist()[0]
        y_predict = list(map(int, y_predict))

        folders = os.listdir(args.val_root)
        folders.sort()  # sort by alphabet
        print("[dataset] class:", folders)
        df_confusion = confusion_matrix(y_actual, y_predict)
        plot_confusion_matrix(df_confusion, folders, args.save_dir + "infer_cm.png", accuracy=best_top1)

    return best_top1, best_top1_name, eval_acces


@torch.no_grad()
def eval_and_save(args, model, val_loader, tlogger):
    tlogger.print("Start Evaluating")
    acc, eval_name, eval_acces = evaluate(args, model, val_loader)
    tlogger.print("....BEST_ACC: {} {}%".format(eval_name, acc))
    ### build records.txt
    msg = "[Evaluation Results]\n"
    msg += "Project: {}, Experiment: {}\n".format(args.project_name, args.exp_name)
    msg += "Samples: {}\n".format(len(val_loader.dataset))
    msg += "\n"
    for name in eval_acces:
        msg += "    {} {}%\n".format(name, eval_acces[name])
    msg += "\n"
    msg += "BEST_ACC: {} {}% ".format(eval_name, acc)

    with open(args.save_dir + "eval_results.txt", "w") as ftxt:
        ftxt.write(msg)


@torch.no_grad()
def test(args, model, test_loader, tlogger, best_eval_name, stage):
    tlogger.print("Start Testing")
    acc, eval_name, eval_acces = evaluate(args, model, test_loader)
    acc = eval_acces[best_eval_name] if best_eval_name in eval_acces else acc
    tlogger.print("....Test_ACC: {} {}%".format(best_eval_name, acc))
    ### build records.txt
    msg = "[Test Results]\n"
    msg += "Project: {}, Experiment: {}\n".format(args.project_name, args.exp_name)
    msg += "Samples: {}\n".format(len(test_loader.dataset))
    msg += "\n"
    for name in eval_acces:
        msg += "    {} {}%\n".format(name, eval_acces[name])
    msg += "\n"
    msg += "Test_ACC: {} {}% ".format(eval_name, acc)

    with open(args.save_dir + "test_results_stage{}.txt".format(stage), "w") as ftxt:
        ftxt.write(msg)


@torch.no_grad()
def eval_and_cm(args, model, val_loader, tlogger):
    tlogger.print("Start Evaluating")
    acc, eval_name, eval_acces = evaluate_cm(args, model, val_loader)
    tlogger.print("....BEST_ACC: {} {}%".format(eval_name, acc))
    ### build records.txt
    msg = "[Evaluation Results]\n"
    msg += "Project: {}, Experiment: {}\n".format(args.project_name, args.exp_name)
    msg += "Samples: {}\n".format(len(val_loader.dataset))
    msg += "\n"
    for name in eval_acces:
        msg += "    {} {}%\n".format(name, eval_acces[name])
    msg += "\n"
    msg += "BEST_ACC: {} {}% ".format(eval_name, acc)

    with open(args.save_dir + "infer_results.txt", "w") as ftxt:
        ftxt.write(msg)


def plot_confusion_matrix(cm, label_names, save_name, title='Confusion Matrix acc = ', accuracy=0):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(len(label_names) / 2, len(label_names) / 2), dpi=100)
    np.set_printoptions(precision=2)
    # print("cm:\n",cm)

    # 统计混淆矩阵中每格的概率值
    x, y = np.meshgrid(np.arange(len(cm)), np.arange(len(cm)))
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        try:
            c = (cm[y_val][x_val] / np.sum(cm, axis=1)[y_val]) * 100
        except KeyError:
            c = 0
        if c > 0.001:
            plt.text(x_val, y_val, "%0.1f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title + str('{:.3f}'.format(accuracy)))
    plt.colorbar()
    plt.xticks(np.arange(len(label_names)), label_names, rotation=45)
    plt.yticks(np.arange(len(label_names)), label_names)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(label_names))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(save_name, format='png')
    # plt.show()


def analyze_prediction_entropy(pred_probs, true_labels, class_names=None,
                              confuse_threshold=0.15, entropy_threshold_low=0.4,
                              entropy_threshold_high=1.1, output_path="entropy_analysis"):
    """
    分析模型预测的熵值，特别针对细粒度分类任务中的易混淆样本

    参数:
    pred_probs (list of np.array): 每个样本的预测概率分布，形状为 [num_samples, num_classes]
    true_labels (list of int): 真实类别标签列表
    class_names (dict or list, optional): 类别ID到名称的映射 {id: 'name'}
    confuse_threshold (float): 定义"易混淆"的错误率阈值 (默认 0.15)
    entropy_threshold_low (float): 低熵阈值 (默认 0.4)
    entropy_threshold_high (float): 高熵阈值 (默认 1.1)
    output_path (str): 输出结果的文件路径前缀

    返回:
    dict: 包含关键分析结果的字典
    """
    # 转换输入为numpy数组
    pred_probs = np.array(pred_probs)
    true_labels = np.array(true_labels)
    num_classes = pred_probs.shape[1]

    # 创建类别名称映射（如果未提供）
    if class_names is None:
        class_names = {i: f"Class_{i}" for i in range(num_classes)}
    elif isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}

    # 1. 计算每个样本的预测熵
    def calculate_entropy(probs):
        # 添加小常数避免log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        return -np.sum(probs * np.log(probs))

    entropies = np.array([calculate_entropy(p) for p in pred_probs])

    # 2. 确定预测类别
    pred_labels = np.argmax(pred_probs, axis=1)

    # 3. 识别错误预测
    is_error = (pred_labels != true_labels)
    error_indices = np.where(is_error)[0]

    # 4. 按混淆对分组分析
    confusion_entropy = defaultdict(lambda: defaultdict(list))
    confusion_count = defaultdict(lambda: defaultdict(int))

    for idx in error_indices:
        true_cls = true_labels[idx]
        pred_cls = pred_labels[idx]
        entropy_val = entropies[idx]

        confusion_entropy[true_cls][pred_cls].append(entropy_val)
        confusion_count[true_cls][pred_cls] += 1

    # 5. 计算每个真实类别的错误率和熵统计
    class_stats = []
    high_priority_pairs = []

    for true_cls in set(true_labels):
        total_samples = np.sum(true_labels == true_cls)
        error_mask = (true_labels == true_cls) & is_error
        error_rate = np.mean(error_mask) if total_samples > 0 else 0

        # 计算该类别的平均熵（仅错误样本）
        cls_error_indices = np.where((true_labels == true_cls) & is_error)[0]
        cls_entropies = entropies[cls_error_indices] if len(cls_error_indices) > 0 else []

        avg_error_entropy = np.mean(cls_entropies) if len(cls_entropies) > 0 else 0
        low_entropy_ratio = np.mean([e < entropy_threshold_low for e in cls_entropies]) if len(cls_entropies) > 0 else 0
        high_entropy_ratio = np.mean([e > entropy_threshold_high for e in cls_entropies]) if len(cls_entropies) > 0 else 0

        # 识别主要混淆目标
        main_confusions = []
        for pred_cls, count in confusion_count[true_cls].items():
            if count / total_samples > 0.05:  # 超过5%的错误指向该类别
                avg_entropy = np.mean(confusion_entropy[true_cls][pred_cls])
                main_confusions.append((pred_cls, count, avg_entropy))

        # 按错误数量排序
        main_confusions.sort(key=lambda x: x[1], reverse=True)

        # 判断是否高优先级（错误率高且样本量足够）
        is_high_priority = (error_rate > confuse_threshold) and (total_samples > 10)

        class_stats.append({
            'class_id': true_cls,
            'class_name': class_names.get(true_cls, f"Class_{true_cls}"),
            'total_samples': total_samples,
            'error_rate': error_rate,
            'avg_error_entropy': avg_error_entropy,
            'low_entropy_ratio': low_entropy_ratio,
            'high_entropy_ratio': high_entropy_ratio,
            'main_confusions': main_confusions,
            'is_high_priority': is_high_priority
        })

        if is_high_priority:
            for pred_cls, _, _ in main_confusions[:3]:  # 取前3个主要混淆
                high_priority_pairs.append((true_cls, pred_cls))

    # 6. 生成分析报告
    print("="*60)
    print("熵值分析报告")
    print("="*60)

    # 全局统计
    total_errors = len(error_indices)
    if total_errors > 0:
        global_avg_entropy = np.mean(entropies[error_indices])
        low_entropy_errors = np.mean(entropies[error_indices] < entropy_threshold_low)
        high_entropy_errors = np.mean(entropies[error_indices] > entropy_threshold_high)

        print(f"全局统计:")
        print(f"- 总错误样本数: {total_errors} ({total_errors/len(true_labels):.1%})")
        print(f"- 错误样本平均熵值: {global_avg_entropy:.3f}")
        print(f"- 低熵错误比例 (<{entropy_threshold_low}): {low_entropy_errors:.1%}")
        print(f"- 高熵错误比例 (>{entropy_threshold_high}): {high_entropy_errors:.1%}")
        print()

    # 高优先级类别分析
    high_priority_classes = [s for s in class_stats if s['is_high_priority']]
    print(f"发现 {len(high_priority_classes)} 个高优先级分析目标 (错误率>{confuse_threshold*100:.0f}%且样本>10):")

    for stat in sorted(high_priority_classes, key=lambda x: (-x['error_rate'], -x['total_samples'])):
        print(f"\n【{stat['class_id']}: {stat['class_name']}】")
        print(f"- 错误率: {stat['error_rate']:.1%} ({stat['total_samples']} 样本)")
        print(f"- 错误样本平均熵: {stat['avg_error_entropy']:.3f}")
        print(f"- 低熵错误比例: {stat['low_entropy_ratio']:.1%} | 高熵错误比例: {stat['high_entropy_ratio']:.1%}")

        if stat['main_confusions']:
            print("- 主要混淆目标:")
            for pred_cls, count, avg_entropy in stat['main_confusions'][:3]:
                ratio = count / stat['total_samples']
                print(f"  → {pred_cls}: {class_names.get(pred_cls, f'Class_{pred_cls}')} "
                      f"({count} 错误, {ratio:.1%} of total, avg entropy={avg_entropy:.3f})")

    # 7. 生成可视化
    plt.figure(figsize=(15, 10))

    # 热力图：按混淆对的平均熵
    confusion_pairs = []
    entropy_values = []
    error_rates = []

    for true_cls, pred_cls_dict in confusion_entropy.items():
        for pred_cls, entropies_list in pred_cls_dict.items():
            if confusion_count[true_cls][pred_cls] > 5:  # 只考虑有足够样本的混淆对
                avg_entropy = np.mean(entropies_list)
                error_rate = confusion_count[true_cls][pred_cls] / np.sum(true_labels == true_cls)

                confusion_pairs.append(f"{true_cls}->{pred_cls}")
                entropy_values.append(avg_entropy)
                error_rates.append(error_rate)

    if confusion_pairs:
        # 创建DataFrame用于绘图
        df = pd.DataFrame({
            'Confusion Pair': confusion_pairs,
            'Average Entropy': entropy_values,
            'Error Rate': error_rates
        })

        # 按错误率排序，取前30个
        df = df.sort_values('Error Rate', ascending=False).head(30)

        # 创建双变量散点图
        plt.subplot(2, 1, 1)
        scatter = plt.scatter(df['Error Rate'], df['Average Entropy'],
                             s=df['Error Rate']*5000, alpha=0.6,
                             c=df['Average Entropy'], cmap='viridis')

        # 添加类别名称标签
        for i, row in df.iterrows():
            true_id = int(row['Confusion Pair'].split('->')[0])
            pred_id = int(row['Confusion Pair'].split('->')[1])
            label = f"{class_names.get(true_id, true_id)}→{class_names.get(pred_id, pred_id)}"
            plt.annotate(label, (row['Error Rate'], row['Average Entropy']),
                        xytext=(5, 2), textcoords='offset points')

        plt.axhline(y=entropy_threshold_low, color='r', linestyle='--', alpha=0.7, label=f'低熵阈值 ({entropy_threshold_low})')
        plt.axhline(y=entropy_threshold_high, color='g', linestyle='--', alpha=0.7, label=f'高熵阈值 ({entropy_threshold_high})')
        plt.xlabel('错误率')
        plt.ylabel('平均熵值')
        plt.title('易混淆对的错误率与熵值关系')
        plt.legend()
        plt.grid(alpha=0.3)

        # 高优先级类别的熵分布
        plt.subplot(2, 1, 2)
        high_priority_ids = [s['class_id'] for s in high_priority_classes]
        if high_priority_ids:
            priority_mask = np.isin(true_labels, high_priority_ids) & is_error
            priority_errors = true_labels[priority_mask]

            # 准备数据
            entropy_data = []
            for idx in np.where(priority_mask)[0]:
                cls = true_labels[idx]
                entropy_val = entropies[idx]
                entropy_data.append({
                    'Class': f"{cls}: {class_names.get(cls, cls)}",
                    'Entropy': entropy_val
                })

            if entropy_data:
                entropy_df = pd.DataFrame(entropy_data)
                plt.axvline(x=entropy_threshold_low, color='r', linestyle='--', alpha=0.7)
                plt.axvline(x=entropy_threshold_high, color='g', linestyle='--', alpha=0.7)
                sns.histplot(data=entropy_df, x='Entropy', hue='Class', kde=True,
                            alpha=0.6, element='step', common_norm=False)
                plt.title('高优先级类别的错误样本熵值分布')
                plt.xlabel('熵值')
                plt.ylabel('样本数')
                plt.grid(alpha=0.3)
            else:
                plt.text(0.5, 0.5, '无高优先级类别的错误样本',
                        ha='center', va='center', transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.5, '无高优先级类别',
                    ha='center', va='center', transform=plt.gca().transAxes)

        plt.tight_layout()
        plt.savefig(f"{output_path}_entropy_analysis.png", dpi=300, bbox_inches='tight')
        print(f"\n已保存熵值分析图表至: {output_path}_entropy_analysis.png")

    # 8. 生成优先级建议
    recommendations = []
    for stat in high_priority_classes:
        if stat['high_entropy_ratio'] > 0.4:  # 高熵错误占比较高
            recommendations.append(
                f"【高潜力】{stat['class_name']} (ID{stat['class_id']}): "
                f"错误率{stat['error_rate']:.1%}，{stat['high_entropy_ratio']:.0%}为高熵错误。"
                "建议：添加地理/物候信息或针对性数据增强，预计可显著提升。"
            )
        elif stat['low_entropy_ratio'] > 0.4:  # 低熵错误占比较高
            recommendations.append(
                f"【高难度】{stat['class_name']} (ID{stat['class_id']}): "
                f"错误率{stat['error_rate']:.1%}，{stat['low_entropy_ratio']:.0%}为低熵错误。"
                "建议：检查模型是否关注错误区域，需重构特征学习或引入领域知识。"
            )
        else:
            recommendations.append(
                f"【中等潜力】{stat['class_name']} (ID{stat['class_id']}): "
                f"错误率{stat['error_rate']:.1%}，熵值分布较均衡。"
                "建议：微调注意力机制或添加同类样本对比学习。"
            )

    if recommendations:
        print("\n" + "="*60)
        print("改进建议")
        print("="*60)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

    # 返回关键结果
    return {
        'global_stats': {
            'total_errors': total_errors,
            'global_avg_entropy': global_avg_entropy if total_errors > 0 else 0,
            'low_entropy_ratio': low_entropy_errors if total_errors > 0 else 0,
            'high_entropy_ratio': high_entropy_errors if total_errors > 0 else 0
        },
        'class_stats': class_stats,
        'high_priority_pairs': high_priority_pairs,
        'recommendations': recommendations
    }

# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    print("示例运行：模拟树种分类任务的熵值分析")

    # 模拟数据（实际使用时替换为您的预测结果）
    np.random.seed(42)
    num_classes = 102
    num_samples = 500

    # 创建类别名称映射（使用您提供的数据）
    class_names = {
        0: "Acer_palmatum", 1: "Aesculus_chinensis", 2: "Ailanthus_altissima",
        3: "Albizia_julibrissin", 4: "Alstonia_scholaris", 5: "Araucaria_cunninghamii",
        6: "Archontophoenix_alexandrae", 7: "Bauhinia_blakeana", 8: "Bauhinia_purpurea",
        9: "Betula_platyphylla", 10: "Bischofia_polycarpa", 11: "Bombax_ceiba",
        12: "Broussonetia_papyrifera", 13: "Caryota_maxima", 14: "Catalpa_ovata",
        15: "Cedrus_deodara", 16: "Ceiba_speciosa", 17: "Celtis_sinensis",
        18: "Celtis_tetrandra", 19: "Cinnamomum_camphora", 20: "Cinnamomum_japonicum",
        21: "Citrus_maxima", 22: "Cocos_nucifera", 23: "Delonix_regia",
        24: "Dimocarpus_longan", 25: "Dracontomelon_duperreanum", 26: "Elaeocarpus_decipiens",
        27: "Elaeocarpus_glabripetalus", 28: "Erythrina_variegata", 29: "Eucommia_ulmoides",
        30: "Euonymus_maackii", 31: "Euphorbia_milii", 32: "Ficus_altissima",
        33: "Ficus_benjamina", 34: "Ficus_concinna", 35: "Ficus_microcarpa",
        36: "Ficus_virens", 37: "Firmiana_simplex", 38: "Fraxinus_chinensis",
        39: "Ginkgo_biloba", 40: "Grevillea_robusta", 41: "Koelreuteria_paniculata",
        42: "Lagerstroemia_indica", 43: "Ligustrum_lucidum", 44: "Ligustrum_quihoui",
        45: "Liquidambar_formosana", 46: "Liriodendron_chinense", 47: "Livistona_chinensis",
        48: "Magnolia_grandiflora", 49: "Mangifera_persiciforma", 50: "Melia_azedarach",
        51: "Metasequoia_glyptostroboides", 52: "Michelia_chapensis", 53: "Morella_rubra",
        54: "Morus_alba", 55: "Osmanthus_fragrans", 56: "Paulownia_tomentosa",
        57: "Phoebe_zhennan", 58: "Photinia_serratifolia", 59: "Picea_asperata",
        60: "Picea_koraiensis", 61: "Picea_meyeri", 62: "Pinus_elliottii",
        63: "Pinus_tabuliformis", 64: "Pinus_thunbergii", 65: "Pittosporum_tobira",
        66: "Platanus", 67: "Platycladus_orientalis", 68: "Populus_alba",
        69: "Populus_canadensis", 70: "Populus_cathayana", 71: "Populus_davidiana",
        72: "Populus_hopeiensis", 73: "Populus_nigra", 74: "Populus_simonii",
        75: "Populus_tomentosa", 76: "Prunus_cerasifera", 77: "Prunus_mandshurica",
        78: "Prunus_salicina", 79: "Pseudolarix_amabilis", 80: "Pterocarya_stenoptera",
        81: "Quercus_robur", 82: "Rhododendron_simsii", 83: "Robinia_pseudoacacia",
        84: "Roystonea_regia", 85: "Sabina_chinensis", 86: "Salix_babylonica",
        87: "Salix_matsudana", 88: "Sapindus_mukorossi", 89: "Sterculia_lanceolata",
        90: "Styphnolobium_japonicum", 91: "Syringa_reticulata_subsp_amurensis",
        92: "Syringa_villosa", 93: "Tilia_amurensis", 94: "Tilia_mandshurica",
        95: "Toona_sinensis", 96: "Trachycarpus_fortunei", 97: "Triadica_sebifera",
        98: "Ulmus_densa", 99: "Ulmus_pumila", 100: "Yulania_denudata", 101: "Zelkova_serrata"
    }

    # 模拟真实标签（根据您提供的数据分布）
    true_labels = []
    for class_id, row in enumerate([
        (0, 54), (1, 72), (2, 46), (3, 42), (4, 55), (5, 28), (6, 74), (7, 78), (8, 19), (9, 24),
        (10, 28), (11, 50), (12, 36), (13, 94), (14, 54), (15, 68), (16, 49), (17, 34), (18, 10), (19, 59),
        (20, 38), (21, 87), (22, 110), (23, 44), (24, 19), (25, 20), (26, 22), (27, 14), (28, 26), (29, 27),
        (30, 54), (31, 68), (32, 35), (33, 44), (34, 10), (35, 114), (36, 28), (37, 35), (38, 18), (39, 79),
        (40, 10), (41, 47), (42, 52), (43, 42), (44, 10), (45, 34), (46, 20), (47, 60), (48, 56), (49, 19),
        (50, 27), (51, 42), (52, 12), (53, 17), (54, 19), (55, 62), (56, 11), (57, 8), (58, 53), (59, 34),
        (60, 16), (61, 39), (62, 12), (63, 26), (64, 10), (65, 46), (66, 35), (67, 42), (68, 14), (69, 14),
        (70, 8), (71, 12), (72, 8), (73, 8), (74, 18), (75, 10), (76, 39), (77, 10), (78, 37), (79, 10),
        (80, 47), (81, 12), (82, 21), (83, 27), (84, 27), (85, 38), (86, 70), (87, 37), (88, 26), (89, 30),
        (90, 44), (91, 51), (92, 10), (93, 10), (94, 10), (95, 10), (96, 88), (97, 64), (98, 8), (99, 46),
        (100, 40), (101, 17)
    ]):
        true_labels.extend([class_id] * row[1])

    # 模拟预测概率（根据您描述的混淆模式）
    pred_probs = []

    for true_cls in true_labels:
        # 根据类别创建不同的预测分布
        probs = np.zeros(num_classes)

        # 设置真实类别的基础概率
        if true_cls == 27:  # Elaeocarpus_glabripetalus - 低熵错误案例
            probs[true_cls] = 0.3  # 真实类别概率较低
            probs[26] = 0.7  # 高概率预测为混淆类别（低熵）
        elif true_cls == 95:  # Toona_sinensis - 高熵错误案例
            probs[true_cls] = 0.4  # 真实类别概率中等
            probs[2] = 0.4  # 混淆类别概率高
            probs[88] = 0.2  # 其他可能类别
        elif true_cls == 59:  # Picea_asperata - 低熵错误案例
            probs[true_cls] = 0.2
            probs[61] = 0.8  # 几乎总是预测为61（低熵）
        elif true_cls == 87:  # Salix_matsudana - 高熵错误案例
            probs[true_cls] = 0.45
            probs[86] = 0.45
            probs[99] = 0.1
        elif true_cls == 34:  # Ficus_concinna - 高熵错误案例
            probs[true_cls] = 0.3
            probs[35] = 0.7
        else:
            # 其他类别：根据错误率生成概率
            error_rate = np.random.beta(1, 5)  # 大部分类别错误率较低
            if np.random.rand() < error_rate:
                # 随机选择一个混淆类别
                confuse_cls = np.random.choice([c for c in range(num_classes) if c != true_cls])
                probs[confuse_cls] = 0.9
                probs[true_cls] = 0.1
            else:
                # 正确预测
                probs[true_cls] = 0.95
                # 分配少量概率给其他类别
                other_clses = np.random.choice([c for c in range(num_classes) if c != true_cls],
                                             size=min(5, num_classes-1), replace=False)
                probs[other_clses] = 0.05 / len(other_clses)

        # 归一化并添加噪声
        probs = probs / np.sum(probs)
        probs = probs + np.random.normal(0, 0.02, num_classes)
        probs = np.clip(probs, 0, 1)
        probs = probs / np.sum(probs)

        pred_probs.append(probs)

    # 运行分析
    results = analyze_prediction_entropy(
        pred_probs=pred_probs,
        true_labels=true_labels,
        class_names=class_names,
        confuse_threshold=0.15,
        entropy_threshold_low=0.4,
        entropy_threshold_high=1.1,
        output_path="tree_species_entropy"
    )

    # 保存详细报告
    with open("entropy_analysis_report.txt", "w") as f:
        f.write("熵值分析详细报告\n")
        f.write("="*50 + "\n\n")

        # 全局统计
        f.write("全局统计:\n")
        f.write(f"总错误样本数: {results['global_stats']['total_errors']}\n")
        f.write(f"错误样本平均熵值: {results['global_stats']['global_avg_entropy']:.3f}\n")
        f.write(f"低熵错误比例: {results['global_stats']['low_entropy_ratio']:.1%}\n")
        f.write(f"高熵错误比例: {results['global_stats']['high_entropy_ratio']:.1%}\n\n")

        # 高优先级类别详情
        f.write("高优先级类别详情:\n")
        for stat in results['class_stats']:
            if stat['is_high_priority']:
                f.write(f"\n{stat['class_id']}: {stat['class_name']}\n")
                f.write(f"错误率: {stat['error_rate']:.1%} | 样本数: {stat['total_samples']}\n")
                f.write(f"平均错误熵: {stat['avg_error_entropy']:.3f}\n")
                f.write(f"低熵错误比例: {stat['low_entropy_ratio']:.1%} | 高熵错误比例: {stat['high_entropy_ratio']:.1%}\n")

                f.write("主要混淆目标:\n")
                for pred_cls, count, avg_entropy in stat['main_confusions'][:3]:
                    f.write(f"  → {pred_cls}({class_names.get(pred_cls, 'N/A')}): "
                           f"{count} 错误, 平均熵={avg_entropy:.3f}\n")

        # 建议
        f.write("\n改进建议:\n")
        for i, rec in enumerate(results['recommendations'], 1):
            f.write(f"{i}. {rec}\n")

    print("\n已保存详细分析报告至: entropy_analysis_report.txt")