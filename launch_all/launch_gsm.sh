#42 base test 0

DEVICES_OF_GSM=0
DATA_DIR="/mnt/data2/mxdi/archive/grade-school-math/grade_school_math/data"
SAVE_DIR="results/gsm/cot-8shot-test"
MODEL_DIR="/mnt/data2/mxdi/models/Llama-2-7b-hf"
SEED="42"

source /mnt/data2/mxdi/archive/open-instruct/launch_all/launch_aaalll.sh

: '
#42 system test 0

DATA_DIR="/mnt/data2/mxdi/archive/grade-school-math/grade_school_math/data"
SAVE_DIR="results/gsm/system-cot-8shot-math"
MODEL_DIR="/mnt/data2/mxdi/ift_prac/math_sys/math-lm_sys"
SEED="42"

#42 no system test 0

DATA_DIR="/mnt/data2/mxdi/archive/grade-school-math/grade_school_math/data"
SAVE_DIR="/mnt/data2/mxdi/archive/open-instruct/results/gsm/nosystem-cot-8shot-math"
MODEL_DIR="/mnt/data2/mxdi/ift_prac/math_sys/math-lm_sys_no"
SEED="42"

#42 base test 0

DATA_DIR="/mnt/data2/mxdi/archive/grade-school-math/grade_school_math/data"
SAVE_DIR="results/gsm/cot-8shot"
MODEL_DIR="/mnt/data2/mxdi/models/Llama-2-7b-hf"
SEED="42"

save：
results/gsm/nosystem-cot-8shot
results/gsm/system-cot-8shot

'