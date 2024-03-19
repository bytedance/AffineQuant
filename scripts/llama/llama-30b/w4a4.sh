CUDA_VISIBLE_DEVICES=0 python main.py \
--model /path/to/llama-30b-hf --eval_ppl --save_dir ./fake_quant_model/llama-30b-w4a4 \
--epochs 20 --output_dir ./log/llama-30b-w4a4 --act-scales ./act_scales/llama-30b.pt --act-shifts ./act_shifts/llama-30b.pt \
--wbits 4 --abits 4 --lwc --let --alpha 0.75 --aug_loss --use_matrix --sf 0.1 \
--tasks hendrycksTest,piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande