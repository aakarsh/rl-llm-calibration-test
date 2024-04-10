#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:..

python  ../llm_calibration/run_experiment.py --model_name='meta-llama/Llama-2-13b-hf' --dataset='STEM' 
python  ../llm_calibration/run_experiment.py --model_name='meta-llama/Llama-2-13b-hf' --dataset='HUMANITIES'
python  ../llm_calibration/run_experiment.py --model_name='meta-llama/Llama-2-13b-hf' --dataset='SOCIAL_SCIENCE' 
python  ../llm_calibration/run_experiment.py --model_name='meta-llama/Llama-2-13b-hf' --dataset='OTHER'

python  ../llm_calibration/run_experiment.py --model_name='meta-llama/Llama-2-13b-chat-hf' --dataset='STEM' 
python  ../llm_calibration/run_experiment.py --model_name='meta-llama/Llama-2-13b-chat-hf' --dataset='HUMANITIES'
python  ../llm_calibration/run_experiment.py --model_name='meta-llama/Llama-2-13b-chat-hf' --dataset='SOCIAL_SCIENCE' 
python  ../llm_calibration/run_experiment.py --model_name='meta-llama/Llama-2-13b--chat-hf' --dataset='OTHER'

python  ../llm_calibration/run_experiment.py --model_name='meta-llama/Llama-2-7b-hf' --dataset='STEM' 
python  ../llm_calibration/run_experiment.py --model_name='meta-llama/Llama-2-7b-hf' --dataset='HUMANITIES'
python  ../llm_calibration/run_experiment.py --model_name='meta-llama/Llama-2-7b-hf' --dataset='SOCIAL_SCIENCE' 
python  ../llm_calibration/run_experiment.py --model_name='meta-llama/Llama-2-7b-hf' --dataset='OTHER'

python  ../llm_calibration/run_experiment.py --model_name='meta-llama/Llama-2-7b-chat-hf' --dataset='STEM' 
python  ../llm_calibration/run_experiment.py --model_name='meta-llama/Llama-2-7b-chat-hf' --dataset='HUMANITIES'
python  ../llm_calibration/run_experiment.py --model_name='meta-llama/Llama-2-7b-chat-hf' --dataset='SOCIAL_SCIENCE' 
python  ../llm_calibration/run_experiment.py --model_name='meta-llama/Llama-2-7b--chat-hf' --dataset='OTHER'
