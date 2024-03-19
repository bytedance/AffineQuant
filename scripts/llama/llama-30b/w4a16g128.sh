CUDA_VISIBLE_DEVICES=0 python main.py \
--model /path/to/llama-30b-hf --eval_ppl --save_dir ./fake_quant_model/llama-30b-w4a16g128 \
--epochs 20 --output_dir ./log/llama-30b-w4a16g128 --act-scales ./act_scales/llama-30b.pt --act-shifts ./act_shifts/llama-30b.pt \
--wbits 4 --abits 16 --group_size 128 --lwc --let --use_ln_matrix --sf 1e-2