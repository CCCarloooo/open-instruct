#改变当前路径
cd /mnt/data2/mxdi/archive/open-instruct/
echo "开始评估gsm8k"

CUDA_VISIBLE_DEVICES=$DEVICES_OF_VLLM python -m eval.gsm.run_eval \
    --data_dir  $GSM_DATA_DIR \
    --max_num_examples 80 \
    --save_dir $GSM_SAVE_DIR \
    --model $MODEL_DIR \
    --tokenizer $MODEL_DIR \
    --eval_batch_size 10 \
    --n_shot 8 \
    --load_in_8bit \
    --seed $SEED \
    --use_vllm \

echo "开始评估mmlu"

CUDA_VISIBLE_DEVICES=$DEVICES_OF_VLLM python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir $MMLU_DATA_DIR \
    --save_dir $MMLU_SAVE_DIR \
    --model_name_or_path $MODEL_DIR\
    --tokenizer_name_or_path $MODEL_DIR\
    --eval_batch_size 16 \
    --load_in_8bit\

echo "开始评估bbh"

CUDA_VISIBLE_DEVICES=$DEVICES_OF_VLLM python -m eval.bbh.run_eval \
    --data_dir $BBH_DATA_DIR \
    --save_dir $BBH_SAVE_DIR \
    --model $MODEL_DIR \
    --tokenizer $MODEL_DIR \
    --eval_batch_size 20 \
    --max_num_examples_per_task 40 \
    --load_in_8bit \
    --use_vllm \

echo "开始评估codex pass@1"

CUDA_VISIBLE_DEVICES=$DEVICES_OF_VLLM python -m eval.codex_humaneval.run_eval \
    --data_file $CODEX_DATA_DIR \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --save_dir $CODEX_SAVE_DIR_a1 \
    --model $MODEL_DIR \
    --tokenizer $MODEL_DIR\
    --eval_batch_size 32 \
    --load_in_8bit \
    --use_vllm

echo "开始评估codex pass@10"

CUDA_VISIBLE_DEVICES=$DEVICES_OF_VLLM python -m eval.codex_humaneval.run_eval \
    --data_file $CODEX_DATA_DIR \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --save_dir $CODEX_SAVE_DIR_a10 \
    --model $MODEL_DIR \
    --tokenizer $MODEL_DIR\
    --eval_batch_size 32 \
    --load_in_8bit\
    --use_vllm

echo "开始评估tydiqa_gold"

CUDA_VISIBLE_DEVICES=$DEVICES_OF_VLLM python -m eval.tydiqa.run_eval \
    --data_dir $TYDIQA_DATA_DIR \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir $TYDIQA_SAVE_DIR \
    --model $MODEL_DIR \
    --tokenizer $MODEL_DIR \
    --eval_batch_size 20 \
    --load_in_8bit \
    --use_vllm

echo "执行结束" 