<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

## ACL'23 DiMS: Distillig Multiple Steps of Iterative Non-Autoregressive Transformer
[[paper](https://arxiv.org/abs/2206.02999)]

Authors: Sajad Norouzi, Rasa Hosseinzadeh, Felipe Pérez, [Maksims Volkovs](http://www.cs.toronto.edu/~mvolkovs)

<a name="intro"/>

## Introduction
This repository contains a full implementation of the DiMS implemented with the fairseq library.

<a name="env"/>

## Environment
The python code is developed and tested on the following environment:
* Python 3.7.6
* Pytorch 1.9.0

Experiments were run on an IBM server with 160 POWER9 CPUs, 600GB RAM and 4 Tesla V100 GPUs

The following command needs to be run in the root of the project before using the repo:
```
pip install -e ./
```

<a name="dataset"/>

## Dataset

For the WMT'14 En-De and WMT'16 En-Ro datasets refer to the fairseq's instructions [here](https://github.com/pytorch/fairseq/tree/master/examples/translation)

## Running The Code

1. The script `./train_cmclc_ende.sh` can be used to train a teacher. The defualt uses 4 GPUS and should
be edited as necessary. The path to dataset should be provided in the first line. The path for checkpoints and logging
should be changed in the script with `--save-dir`, `--tensorboard-logdir` and `--log-file`. Note that the provided
directories should exist before running the script.
2. To distill use `exp_manager.py`. Example settings are provided in ExpSetting directory. These scripts
should be edited to containt correct path to dataset and teacher checkpoints. Run like
```python exp_manager.py cmlmc_ende 0,1,2,3```
,where cmlmc_ende.json is inside ExpSetting directory.
3. To evaluate any model using test set run `./eval_teacher_wmt.sh`. The arguments are as follows:
```
./eval_teacher_wmt.sh PATH_TO_MODEL PATH_TO_DATA NUMBER_OF_STEPS LENGTH_PREDICTOR_BEAM [--ctc]
```
where --ctc is optional for evaluating Imputer models.
