CUDA_VISIBLE_DEVICES=0 python main.py \
--model /path/to/opt-1.3b --eval_ppl --save_dir ./fake_quant_model/opt-1.3b-w2a16g64 \
--epochs 40 --output_dir ./log/opt-1.3b-w2a16g64 --act-scales ./act_scales/opt-1.3b.pt --act-shifts ./act_shifts/opt-1.3b.pt \
--wbits 2 --abits 16 --group_size 64 --lwc --let --use_ln_matrix --sf 0.1