#!/bin/bash
python convert_nezha_original_tf_checkpoint_to_pytorch.py \
	    --tf_checkpoint_path=./nezha-cn-wwm/model.ckpt \
	    --nezha_config_file=./nezha-cn-wwm/bert_config.json \
	    --pytorch_dump_path=./nezha-cn-wwm/pytorch_model.bin
python preprocess.py
python save_embeddings.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 nezha_base_distribute_pretrain.py --fp16
CUDA_VISIBLE_DEVIDES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 nezha_wwm_distribute_pretrain.py --fp16 --seed 2021
CUDA_VISIBLE_DEVIDES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 roberta_distribute_pretrain.py --fp16 --seed 10590 --epoch 150
CUDA_VISIBLE_DEVIDES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 roberta_uer_distribute_pretrain.py --fp16 --seed 100 --epoch 150
for n in $(seq 0 3);
do
    if [ $n -eq 0 ]
        then
	    CUDA_VISIBLE_DEVICES=0 python nezha_base_distribute_finetune.py --do_train --struc cls --epoch 3 &
    elif [ $n -eq 1 ]
        then
	    CUDA_VISIBLE_DEVICES=1 python nezha_wwm_distribute_finetune.py --do_train --struc cls --epoch 3 --seed 2021 &
    elif [ $n -eq 2 ]
	then
	    CUDA_VISIBLE_DEVICES=2 python roberta_distribute_finetune.py --do_train --struc cls --epoch 3 --seed 10590 &
    elif [ $n -eq 3 ]
	then
            CUDA_VISIBLE_DEVICES=3 python roberta_uer_distribute_finetune.py --do_train --struc cls --epoch 3 --seed 100 &
    fi
done
wait
python distribute_to_onnx.py --struc cls --model_name nezha_base --model_path ./finetune_model/base_cls_best_model.pth --filepath best_model1.onnx
python distribute_to_onnx.py --struc cls --model_name nezha_wwm --model_path ./finetune_model/wwm_cls_best_model.pth --filepath best_model2.onnx --seed 2021
python roberta_distribute_to_onnx.py --struc cls --model_name hgf --model_path ./finetune_model/roberta_cls_best_model.pth --filepath best_model3.onnx --seed 10590
python roberta_distribute_to_onnx.py --struc cls --model_name uer --model_path ./finetune_model/uer_cls_best_model.pth --filepath best_model4.onnx --seed 100
python merge_main.py
