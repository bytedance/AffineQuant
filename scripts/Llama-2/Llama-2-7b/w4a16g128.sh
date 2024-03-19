CUDA_VISIBLE_DEVICES=0 python main.py \
--model /path/to/Llama-2-7b-hf --eval_ppl --save_dir ./fake_quant_model/Llama-2-7b-w4a16g128 \
--epochs 20 --output_dir ./log/Llama-2-7b-w4a16g128 \
--wbits 4 --abits 16 --group_size 128 --lwc --let --tasks hendrycksTest,piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande