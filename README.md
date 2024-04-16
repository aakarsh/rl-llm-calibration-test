# LLM Calibration Benchmark.


This repository attempts to run benchmarks on some popular openly available 
language models.

## Installation

```
pip install -r requirements. tx
```

When running in the colab environment it is recommended ot use. 

```
pip install -r requirements-colab.txt
```

## Unit Tests

Running unit tests requires pytest module invoked as follows: 

```
    python -m pytest test
```

## Running Individual Experiments 

Any individual experiment can be rerun using the following command

```
python  ../llm_calibration/run_experiment.py --model_name='meta-llama/Llama-2-13b-hf' --dataset='STEM' 
```

The experimental result will produce a json result files which can be parsed offline to generate the requisite plots.  
