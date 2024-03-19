CUDA_VISIBLE_DEVICES=0 python main.py \
--model /path/to/Llama-2-7b-hf --eval_ppl --save_dir ./fake_quant_model/Llama-2-7b-w3a16 \
--epochs 20 --output_dir ./log/Llama-2-7b-w3a16 --act-scales ./act_scales/Llama-2-7b.pt --act-shifts ./act_shifts/Llama-2-7b.pt \
--wbits 3 --abits 16 --lwc --let --use_ln_matrix --sf 1e-2