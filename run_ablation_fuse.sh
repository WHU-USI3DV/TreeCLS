#/usr/bin
CUDA_VISIBLE_DEVICES=0 python main.py --c configs/whole_pipeline.yaml --f 1 --n ablation_logits_mlp
CUDA_VISIBLE_DEVICES=0 python main.py --c configs/whole_pipeline.yaml --f 3 --n ablation_logits_attn
CUDA_VISIBLE_DEVICES=0 python main.py --c configs/whole_pipeline.yaml --f 2 --n ablation_feature_mlp
CUDA_VISIBLE_DEVICES=0 python main.py --c configs/whole_pipeline.yaml --f 4 --n ablation_feature_attn
CUDA_VISIBLE_DEVICES=0 python eval.py --pr records/WHU/ablation_logits_mlp --f 1
CUDA_VISIBLE_DEVICES=0 python eval.py --pr records/WHU/ablation_logits_attn --f 3
CUDA_VISIBLE_DEVICES=0 python eval.py --pr records/WHU/ablation_feature_mlp --f 2
CUDA_VISIBLE_DEVICES=0 python eval.py --pr records/WHU/ablation_feature_attn --f 4