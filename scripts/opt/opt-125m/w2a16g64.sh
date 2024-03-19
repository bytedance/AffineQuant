CUDA_VISIBLE_DEVICES=0 python main.py \
--model /path/to/opt-125m --eval_ppl --save_dir ./fake_quant_model/opt-125m-w2a16g64 \
--epochs 40 --output_dir ./log/opt-125m-w2a16g64 --act-scales ./act_scales/opt-125m.pt --act-shifts ./act_shifts/opt-125m.pt \
--wbits 2 --abits 16 --group_size 64 --lwc --let --use_ln_matrix --sf 1.0