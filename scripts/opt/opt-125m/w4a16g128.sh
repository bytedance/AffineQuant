CUDA_VISIBLE_DEVICES=0 python main.py \
--model /path/to/opt-125m --eval_ppl --save_dir ./fake_quant_model/opt-125m-w4a16g128 \
--epochs 20 --output_dir ./log/opt-125m-w4a16g128 --act-scales ./act_scales/opt-125m.pt --act-shifts ./act_shifts/opt-125m.pt \
--wbits 4 --abits 16 --group_size 128 --lwc --let --use_ln_matrix --sf 1.0