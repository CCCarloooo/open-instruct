MODEL_DIR="/mnt/data2/mxdi/archive/models/cl_llava_vicuna/4000_7b"
MODEL_NAME="4000_7b_llava-7b-vicuna"
SEED="42"
DEVICES_OF_VLLM=0,1


GSM_DATA_DIR="/mnt/data2/mxdi/archive/grade-school-math/grade_school_math/data"
GSM_SAVE_DIR="results/$MODEL_NAME/gsm/test"

#mmlu
MMLU_DATA_DIR="data/eval/mmlu"
MMLU_SAVE_DIR="results/$MODEL_NAME/mmlu/test"

#bbh
BBH_DATA_DIR="data/eval/bbh"
BBH_SAVE_DIR="results/$MODEL_NAME/bbh/test"

#code
CODEX_DATA_DIR="data/eval/codex_humaneval/HumanEval.jsonl.gz"
CODEX_SAVE_DIR_a1="results/$MODEL_NAME/codex_humaneval/test/a1"
CODEX_SAVE_DIR_a10="results/$MODEL_NAME/codex_humaneval/test/a10"

#tydiqa
TYDIQA_DATA_DIR="data/eval/tydiqa"
TYDIQA_SAVE_DIR="results/$MODEL_NAME/tydiqa/test"


source /mnt/data2/mxdi/archive/open-instruct/launch_all/launch_aaalll.sh