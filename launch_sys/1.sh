DATA_DIR="/mnt/data2/mxdi/archive/grade-school-math/grade_school_math/data"
SEED="23"

SAVE_DIR="results/gsm/llava-8shot-testvllm"
MODEL_DIR="/mnt/data2/mxdi/models/llava-vicuna"
TOKENIZER_DIR="/mnt/data2/mxdi/models/llava-vicuna"


source /mnt/data2/mxdi/archive/open-instruct/launch_all/system_prompt_eval.sh

: '
"/mnt/data2/mxdi/archive/hf-mirror/vicuna-7b-v1.5"
"/mnt/data2/mxdi/models/llava-vicuna"

'