echo "开始执行"
#改变当前路径
cd /mnt/data2/mxdi/archive/open-instruct/

# llava cot
# By default, we use 1-shot setting, and 100 examples per language
CUDA_VISIBLE_DEVICES=0,1 python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir results/tydiqa/vicuna-goldp-vllm \
    --model /mnt/data2/mxdi/archive/hf-mirror/vicuna-7b-v1.5 \
    --tokenizer /mnt/data2/mxdi/archive/hf-mirror/vicuna-7b-v1.5 \
    --eval_batch_size 20 \
    --load_in_8bit \
    --use_vllm

echo "执行结束" 

: '
# Evaluating with gold passage provided
# By default, we use 1-shot setting, and 100 examples per language
CUDA_VISIBLE_DEVICES=0,1,2 python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir results/tydiqa/llava-goldp \
    --model /mnt/data2/mxdi/models/llava-vicuna \
    --tokenizer /mnt/data2/mxdi/models/llava-vicuna \
    --eval_batch_size 20 \
    --load_in_8bit


# Evaluating with no context provided (closed-book QA)
# By default, we use 1-shot setting, and 100 examples per language
CUDA_VISIBLE_DEVICES=0,1,2 python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir results/tydiqa/llava-no-context \
    --model /mnt/data2/mxdi/models/llava-vicuna\
    --tokenizer /mnt/data2/mxdi/models/llava-vicuna \
    --eval_batch_size 40 \
    --load_in_8bit \
    --no_context   

# vicuna with gold passage provided
# By default, we use 1-shot setting, and 100 examples per language
CUDA_VISIBLE_DEVICES=0,1,2 python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir results/tydiqa/vicuna-goldp \
    --model /mnt/data2/mxdi/archive/hf-mirror/vicuna-7b-v1.5 \
    --tokenizer /mnt/data2/mxdi/archive/hf-mirror/vicuna-7b-v1.5 \
    --eval_batch_size 20 \
    --load_in_8bit
'