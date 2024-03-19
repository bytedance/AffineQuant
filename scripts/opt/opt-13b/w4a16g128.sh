CUDA_VISIBLE_DEVICES=0 python main.py \
--model /path/to/opt-13b --eval_ppl --save_dir ./fake_quant_model/opt-13b-w4a16g128 \
--epochs 20 --output_dir ./log/opt-13b-w4a16g128 --act-scales ./act_scales/opt-13b.pt --act-shifts ./act_shifts/opt-13b.pt \
--wbits 4 --abits 16 --group_size 128 --lwc --let --use_ln_matrix --sf 1e-2