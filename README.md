# NLPCtlScalingAndBounds

[Public repository]

Experiment codes for the paper: Explaining Context Length Scaling and Bounds for Language Models.

## Content in this repository

The repository contains three major parts of code.

1. Synthetic Data:

	generating data, training models, measuring Cross Entropy Loss, obtaining middle-layer feature representation of our Synthetic Dataset

2. Measuring Natural Language:

	generating text corpera, measuring CE Loss, obtaining middle-layer feature representation

3. Experiments on Openwebtext subset:

	generating training sub-dataset, training gpt-2 with nanogpt



## Part 1. Synthetic Dataset

### 1.1. Environment setting-up

Please refer to `SyntheticDataset/requirements.txt` for requirements.

### 1.2. Generating Synthetic Dataset

Run `generate_data.py`, which would generate training/validation set according to the task defined in `task_definition.json`

### 1.3. Train MLP, obtain Validation Loss v.s. Context Length v.s. Training dataset size

Run `train_model.sh` with different context length, dataset sizes settings.

### 1.4. Train MLP with specialized architecture, then obtain context feature for PCA.

Run `train_model.py` `--use_bi_mlp` setting, and train Bi-mlp on different context length. Then, run `save_feature_tensor.py` with the corresponding model weight to obtain middle features.

## Part 2. Measuring Natural Language with LLaMa

### 2.1. Environment setting-up

Please refer to `NaturalLanguaage/Measuring/requirements.txt` for requirements.

### 2.2. Prepare dataset

Please refer to `prepare_data.py` for preparing datasets.

### 2.3. Measuring LLaMa CE loss

Run `evaluate_CE.py` with different `seq_lens` and `model_name` settings to obtain results.

### 2.4. Saving middle-tensors for PCA analysis

Run `save_mid_features.py` to obtain different feature vectors.

### 2.5. Draw CE v.s. ID, or CE v.s. sum(log(eigval))

Please refer to python scripts in `NaturalLanguaage\Measuring\draw_CEvsPCA` for drawing figures.

## Part 3. Training and Evaluating on Openwebtext

### 3.1. Environment setting-up

Please refer to `NaturalLanguaage/ContextLengthScalingTrainingExps/nanoGPT-master/README.md` for the installation of nanogpt.

### 3.2. Generating dataset

Run `data/openwebtext/prepare.py` with `percent` and `start_from` set to a appropriate value.

### 3.3. Train models.

Run `script.py` to generate config files (already generated now). Then, run `sbatch train_script.sh 5120 2p0` to start training for context length 5120, dataset percent 2, on clusters we use. Please modify the script to run jobs on other machines.