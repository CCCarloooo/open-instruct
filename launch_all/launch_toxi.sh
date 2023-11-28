echo "开始执行"
#改变当前路径
cd /mnt/data2/mxdi/archive/open-instruct/

CUDA_VISIBLE_DEVICES=3 python -m eval.toxigen.run_eval \
    --data_dir data/eval/toxigen/ \
    --save_dir data/eval/toxigen/llava  \
    --model_name_or_path /mnt/data2/mxdi/models/llava-vicuna \
    --tokenizer_name_or_path /mnt/data2/mxdi/models/llava-vicuna \
    --eval_batch_size 32 \
    --use_vllm

echo "执行结束" 
: '


'