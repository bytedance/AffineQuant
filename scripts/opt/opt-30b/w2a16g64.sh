CUDA_VISIBLE_DEVICES=0 python main.py \
--model /path/to/opt-30b --eval_ppl --save_dir ./fake_quant_model/opt-30b-w2a16g64 \
--epochs 40 --output_dir ./log/opt-30b-w2a16g64 --act-scales ./act_scales/opt-30b.pt --act-shifts ./act_shifts/opt-30b.pt \
--wbits 2 --abits 16 --group_size 64 --lwc --let --use_ln_matrix --sf 1e-2