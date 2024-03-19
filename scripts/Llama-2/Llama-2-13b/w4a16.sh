CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/Llama-2-13b-hf --eval_ppl --save_dir ./fake_quant_model/Llama-2-13b-w4a16 \
--epochs 20 --output_dir ./log/Llama-2-13b-w4a16 \
--wbits 4 --abits 16 --lwc --let --tasks hendrycksTest,piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande