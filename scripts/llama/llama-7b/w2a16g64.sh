CUDA_VISIBLE_DEVICES=0 python main.py \
--model /path/to/llama-7b-hf --eval_ppl --save_dir ./fake_quant_model/llama-7b-w2a16g64 \
--epochs 40 --output_dir ./log/llama-7b-w2a16g64 --act-scales ./act_scales/llama-7b.pt --act-shifts ./act_shifts/llama-7b.pt \
--wbits 2 --abits 16 --group_size 64 --lwc --aug_loss --let --use_ln_matrix --sf 1e-2