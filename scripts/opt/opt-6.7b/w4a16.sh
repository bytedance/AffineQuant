CUDA_VISIBLE_DEVICES=0 python main.py \
--model /path/to/opt-6.7b --eval_ppl --save_dir ./fake_quant_model/opt-6.7b-w4a16 \
--epochs 20 --output_dir ./log/opt-6.7b-w4a16 --act-scales ./act_scales/opt-6.7b.pt --act-shifts ./act_shifts/opt-6.7b.pt \
--wbits 4 --abits 16 --lwc --let --use_ln_matrix --sf 1.0