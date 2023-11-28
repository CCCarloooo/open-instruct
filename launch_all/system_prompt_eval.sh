
echo "开始执行"
#改变当前路径
cd /mnt/data2/mxdi/archive/open-instruct/

CUDA_VISIBLE_DEVICES=4,5 python -m eval.gsm.run_eval \
    --data_dir $DATA_DIR \
    --max_num_examples 80 \
    --save_dir $SAVE_DIR \
    --model $MODEL_DIR\
    --tokenizer $MODEL_DIR \
    --eval_batch_size 10 \
    --n_shot 8 \
    --load_in_8bit \
    --seed $SEED \
    --use_vllm \

echo "执行结束" 
: '
测试使用测试集的情况，exact_macth分数太低，仅仅0.025-0.035
python -m eval.gsm.run_eval \
    --data_dir /mnt/data2/mxdi/archive/grade-school-math/grade_school_math/data \
    --max_num_examples 200 \
    --save_dir results/gsm/system-cot-8shot \
    --model /mnt/data2/mxdi/ift_prac/math_sys/math-lm_sys\
    --tokenizer /mnt/data2/mxdi/ift_prac/math_sys/math-lm_sys \
    --eval_batch_size 20 \
    --n_shot 8 \
    --load_in_8bit \

测试没有加入system的情况
python -m eval.gsm.run_eval \
    --data_dir /mnt/data2/mxdi/archive/grade-school-math/grade_school_math/data \
    --max_num_examples 200 \
    --save_dir results/gsm/nosystem-cot-8shot \
    --model MODEL_DIR\
    --tokenizer MODEL_DIR \
    --eval_batch_size 20 \
    --n_shot 8 \
    --load_in_8bit \
测试使用system Prompt 以及 test数据集
python -m eval.gsm.run_eval \
    --data_dir /mnt/data2/mxdi/archive/grade-school-math/grade_school_math/data \
    --max_num_examples 200 \
    --save_dir results/gsm/test_system-cot-8shot \
    --model /mnt/data2/mxdi/ift_prac/math_sys/math-lm_sys\
    --tokenizer /mnt/data2/mxdi/ift_prac/math_sys/math-lm_sys \
    --eval_batch_size 20 \
    --n_shot 8 \
    --load_in_8bit \

'