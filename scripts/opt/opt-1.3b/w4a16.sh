CUDA_VISIBLE_DEVICES=0 python main.py \
--model /path/to/opt-1.3b --eval_ppl --save_dir ./fake_quant_model/opt-1.3b-w4a16 \
--epochs 20 --output_dir ./log/opt-1.3b-w4a16 --act-scales ./act_scales/opt-1.3b.pt --act-shifts ./act_shifts/opt-1.3b.pt \
--wbits 4 --abits 16 --lwc --let --use_ln_matrix --sf 0.1