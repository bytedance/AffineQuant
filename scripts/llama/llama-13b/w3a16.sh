CUDA_VISIBLE_DEVICES=0 python main.py \
--model /path/to/llama-13b-hf --eval_ppl --save_dir ./fake_quant_model/llama-13b-w3a16 \
--epochs 20 --output_dir ./log/llama-13b-w3a16 --act-scales ./act_scales/llama-13b.pt --act-shifts ./act_shifts/llama-13b.pt \
--wbits 3 --abits 16 --lwc --let --use_ln_matrix --sf 1e-2