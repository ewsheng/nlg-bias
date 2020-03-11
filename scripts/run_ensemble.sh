#!/usr/bin/env bash

# Modify params.
export DATA_DIR=data/generated_samples
export TEST_BASE=${2}

# Fixed params.
export MAX_LENGTH=128
export REGARD1_OUTPUT_DIR=models/bert_regard_v1
export REGARD2_OUTPUT_DIR=models/bert_regard_v2
export SENTIMENT1_OUTPUT_DIR=models/bert_sentiment_v1
export SENTIMENT2_OUTPUT_DIR=models/bert_sentiment_v2
export TEST_FILE=${TEST_BASE}.tsv.XYZ

if [[ ${1} == "regard2" ]]
then
    export OUTPUT_DIR=${REGARD2_OUTPUT_DIR}
    export BERT_MODEL1=${OUTPUT_DIR}/checkpoint-90
    export BERT_MODEL2=${OUTPUT_DIR}_2/checkpoint-90
    export BERT_MODEL3=${OUTPUT_DIR}_3/checkpoint-60
    export MODEL_VERSION=2
elif [[ ${1} == "sentiment2" ]]
then
    export OUTPUT_DIR=${SENTIMENT2_OUTPUT_DIR}
    export BERT_MODEL1=${OUTPUT_DIR}/checkpoint-60
    export BERT_MODEL2=${OUTPUT_DIR}_2/checkpoint-60
    export BERT_MODEL3=${OUTPUT_DIR}_3/checkpoint-40
    export MODEL_VERSION=2
elif [[ ${1} == "regard1" ]]
then
    export OUTPUT_DIR=${REGARD1_OUTPUT_DIR}
    export BERT_MODEL1=${OUTPUT_DIR}/checkpoint-40
    export BERT_MODEL2=${OUTPUT_DIR}_2/checkpoint-40
    export BERT_MODEL3=${OUTPUT_DIR}_3/checkpoint-40
    export MODEL_VERSION=1
elif [[ ${1} == "sentiment1" ]]
then
    export OUTPUT_DIR=${SENTIMENT1_OUTPUT_DIR}
    export BERT_MODEL1=${OUTPUT_DIR}/checkpoint-30
    export BERT_MODEL2=${OUTPUT_DIR}_2/checkpoint-40
    export BERT_MODEL3=${OUTPUT_DIR}_3/checkpoint-50
    export MODEL_VERSION=1
fi
export ENSEMBLE_DIR=${OUTPUT_DIR}/generated_data_ensemble
export OUTPUT_PREFIX=${DATA_DIR}/${1}_${TEST_BASE}.tsv

echo "Labeling with first classifier..."
python scripts/run_classifier.py --data_dir ${DATA_DIR} \
--model_type bert \
--model_name_or_path ${BERT_MODEL1} \
--output_dir ${OUTPUT_DIR} \
--max_seq_length  ${MAX_LENGTH} \
--do_predict \
--test_file ${TEST_FILE} \
--do_lower_case \
--overwrite_cache \
--per_gpu_eval_batch_size 32 \
--model_version ${MODEL_VERSION}

echo "Labeling with second classifier..."
python scripts/run_classifier.py --data_dir ${DATA_DIR} \
--model_type bert \
--model_name_or_path ${BERT_MODEL2} \
--output_dir ${OUTPUT_DIR}_2 \
--max_seq_length  ${MAX_LENGTH} \
--do_predict \
--test_file ${TEST_FILE} \
--do_lower_case \
--overwrite_cache \
--per_gpu_eval_batch_size 32 \
--model_version ${MODEL_VERSION}

echo "Labeling with third classifier..."
python scripts/run_classifier.py --data_dir ${DATA_DIR} \
--model_type bert \
--model_name_or_path ${BERT_MODEL3} \
--output_dir ${OUTPUT_DIR}_3 \
--max_seq_length  ${MAX_LENGTH} \
--do_predict \
--test_file ${TEST_FILE} \
--do_lower_case \
--overwrite_cache \
--per_gpu_eval_batch_size 32 \
--model_version ${MODEL_VERSION}

echo "Collecting majority labels..."
mkdir -p ${ENSEMBLE_DIR}
cp ${OUTPUT_DIR}/${TEST_BASE}_predictions.txt ${ENSEMBLE_DIR}/1.txt
cp ${OUTPUT_DIR}_2/${TEST_BASE}_predictions.txt ${ENSEMBLE_DIR}/2.txt
cp ${OUTPUT_DIR}_3/${TEST_BASE}_predictions.txt ${ENSEMBLE_DIR}/3.txt
python scripts/ensemble.py --data_dir ${ENSEMBLE_DIR} --output_prefix ${OUTPUT_PREFIX} --file_with_demographics ${DATA_DIR}/${TEST_BASE}.tsv

echo "Done!"
