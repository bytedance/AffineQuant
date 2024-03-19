CUDA_VISIBLE_DEVICES=0 python main.py \
--model /path/to/Llama-2-7b-hf --eval_ppl --save_dir ./fake_quant_model/Llama-2-7b-w2a16g64 \
--epochs 40 --output_dir ./log/Llama-2-7b-w2a16g64 --act-scales ./act_scales/Llama-2-7b.pt --act-shifts ./act_shifts/Llama-2-7b.pt \
--wbits 2 --abits 16 --group_size 64 --lwc --aug_loss --let --use_ln_matrix --sf 1e-2