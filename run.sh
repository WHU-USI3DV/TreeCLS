#/usr/bin
CUDA_VISIBLE_DEVICES=0 python main.py --c configs/whole_pipeline.yaml --b 8 --n whole_pipeline
CUDA_VISIBLE_DEVICES=0 python main.py --c configs/whole_pipeline.yaml --b 2 --n ablation_k=2
CUDA_VISIBLE_DEVICES=0 python main.py --c configs/whole_pipeline.yaml --b 4 --n ablation_k=4
CUDA_VISIBLE_DEVICES=0 python main.py --c configs/whole_pipeline.yaml --b 16 --n ablation_k=16
CUDA_VISIBLE_DEVICES=0 python main.py --c configs/whole_pipeline.yaml --b 32 --n ablation_k=32
