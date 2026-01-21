#/usr/bin
CUDA_VISIBLE_DEVICES=0 python main.py --c configs/no_all.yaml --m 3 --n vit_base
CUDA_VISIBLE_DEVICES=0 python main.py --c configs/whole_pipeline.yaml --m 3 --n vit_base_with_ours
CUDA_VISIBLE_DEVICES=0 python main.py --c configs/no_all.yaml --m 1 --n resnet50
CUDA_VISIBLE_DEVICES=0 python main.py --c configs/whole_pipeline.yaml --m 1 --n resnet50_with_ours
CUDA_VISIBLE_DEVICES=0 python main.py --c configs/no_all.yaml --m 2 --n swint
CUDA_VISIBLE_DEVICES=0 python main.py --c configs/whole_pipeline.yaml --m 2 --n swint_with_ours
CUDA_VISIBLE_DEVICES=0 python main.py --c configs/no_all.yaml --m 4 --n finetune_bioclip
CUDA_VISIBLE_DEVICES=0 python eval.py --pr records/WHU/finetune_bioclip --m 4
CUDA_VISIBLE_DEVICES=0 python eval.py --pr records/WHU/vit_base --m 3
CUDA_VISIBLE_DEVICES=0 python eval.py --pr records/WHU/vit_base_with_ours --m 3
CUDA_VISIBLE_DEVICES=0 python eval.py --pr records/WHU/swint_with_ours --m 2
CUDA_VISIBLE_DEVICES=0 python eval.py --pr records/WHU/resnet50 --m 1
CUDA_VISIBLE_DEVICES=0 python eval.py --pr records/WHU/resnet50_with_ours --m 1
CUDA_VISIBLE_DEVICES=0 python eval.py --pr records/WHU/swint --m 2
