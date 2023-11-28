echo "开始执行"
#改变当前路径
cd /mnt/data2/mxdi/archive/open-instruct/

CUDA_VISIBLE_DEVICES=4,5 python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/llava-7b \
    --model_name_or_path /mnt/data2/mxdi/models/llava-7b-origin-vicuna\
    --tokenizer_name_or_path /mnt/data2/mxdi/models/llava-7b-origin-vicuna\
    --eval_batch_size 16 \
    --load_in_8bit\


echo "执行结束" 
: '
#use format
CUDA_VISIBLE_DEVICES=3,4,5 python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/vicuna-7b-5shot-chat \
    --model_name_or_path /mnt/data2/mxdi/archive/hf-mirror/vicuna-7b-v1.5\
    --tokenizer_name_or_path /mnt/data2/mxdi/archive/hf-mirror/vicuna-7b-v1.5\
    --eval_batch_size 4 \
    --load_in_8bit\
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_vicuna_chat_format
'