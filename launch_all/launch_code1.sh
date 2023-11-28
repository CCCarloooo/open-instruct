echo "开始执行"
#改变当前路径
cd /mnt/data2/mxdi/archive/open-instruct/

CUDA_VISIBLE_DEVICES=4,5 python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --save_dir results/codex_humaneval/vicuna_temp_0_1_vllm-1device \
    --model /mnt/data2/mxdi/archive/hf-mirror/vicuna-7b-v1.5 \
    --tokenizer /mnt/data2/mxdi/archive/hf-mirror/vicuna-7b-v1.5\
    --eval_batch_size 32 \
    --load_in_8bit \
    --use_vllm


echo "执行结束" 
: '
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --save_dir results/codex_humaneval/vicuna_temp_0_1_vllm \
    --model /mnt/data2/mxdi/archive/hf-mirror/vicuna-7b-v1.5 \
    --tokenizer /mnt/data2/mxdi/archive/hf-mirror/vicuna-7b-v1.5\
    --eval_batch_size 32 \
    --load_in_8bit \
    --use_vllm
'