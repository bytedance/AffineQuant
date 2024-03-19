CUDA_VISIBLE_DEVICES=0 python main.py \
--model /path/to/Llama-2-7b-hf --eval_ppl --save_dir ./fake_quant_model/Llama-2-7b-w4a4 \
--epochs 20 --output_dir ./log/Llama-2-7b-w4a4 --act-scales ./act_scales/Llama-2-7b.pt --act-shifts ./act_shifts/Llama-2-7b.pt \
--wbits 4 --abits 4 --lwc --let --use_matrix --sf 0.1 --let_lr 1e-3 --alpha 0.75 \
--tasks hendrycksTest,piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande