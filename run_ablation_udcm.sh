CUDA_VISIBLE_DEVICES=0 python main.py --c configs/whole_pipeline.yaml --f 5 --n ablation_temp_norm
CUDA_VISIBLE_DEVICES=0 python main.py --c configs/whole_pipeline.yaml --f 6 --n ablation_no_bin_center
CUDA_VISIBLE_DEVICES=0 python main.py --c configs/whole_pipeline.yaml --f 7 --n ablation_only_class_uncertainty
CUDA_VISIBLE_DEVICES=0 python main.py --c configs/whole_pipeline.yaml --f 8 --n ablation_only_instance_uncertainty
CUDA_VISIBLE_DEVICES=0 python eval.py --pr records/WHU/ablation_temp_norm --f 5
CUDA_VISIBLE_DEVICES=0 python eval.py --pr records/WHU/ablation_no_bin_center --f 6
CUDA_VISIBLE_DEVICES=0 python eval.py --pr records/WHU/ablation_only_class_uncertainty --f 7
CUDA_VISIBLE_DEVICES=0 python eval.py --pr records/WHU/ablation_only_instance_uncertainty --f 8