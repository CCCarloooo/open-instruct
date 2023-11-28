echo "开始执行"
#改变当前路径
cd /mnt/data2/mxdi/archive/open-instruct/

CUDA_VISIBLE_DEVICES=0 python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --save_dir results/codex_humaneval/llava_temp_0_1_vllm-test \
    --model /mnt/data2/mxdi/models/llava-vicuna \
    --tokenizer /mnt/data2/mxdi/models/llava-vicuna\
    --eval_batch_size 32 \
    --load_in_8bit \
    --use_vllm

echo "执行结束" 
: '
# Evaluating llava model using temperature 0.1 to get the pass@1 score
CUDA_VISIBLE_DEVICES=3,4,5 python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --save_dir results/codex_humaneval/llava_temp_0_1 \
    --model /mnt/data2/mxdi/models/llava-vicuna \
    --tokenizer /mnt/data2/mxdi/models/llava-vicuna\
    --eval_batch_size 32 \
    --load_in_8bit


# Evaluating llama 7B model using temperature 0.8 to get the pass@10 score
CUDA_VISIBLE_DEVICES=3,4,5 python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --save_dir results/codex_humaneval/llava_temp_0_8 \
    --model /mnt/data2/mxdi/models/llava-vicuna \
    --tokenizer /mnt/data2/mxdi/models/llava-vicuna\
    --eval_batch_size 32 \
    --load_in_8bit
'