## Code for paper: **Explaining Context Length Scaling and Bounds for Language Models** [[arXiv Link](https://arxiv.org/abs/2502.01481)]

**Authors:** [Jingzhe Shi](jingzheshi.github.io)$^{1,2\ eq}$, [Qinwei Ma](https://aquahorsem.github.io/)$^{1\ eq}$, [Hongyi Liu](https://www.linkedin.com/in/hongyi-liu-1442332b1)$^{3\ eq}$, [Hang Zhao](http://www.mit.edu/~hangzhao/)$^{1\ \star}$, [Jenq-Neng Hwang](https://people.ece.uw.edu/hwang/)$^{4}$,  [Lei Li](https://llei66.github.io/li-lei.github.io/)$^{4,5\ \star}$.

&emsp; $^{eq}$: equal contribution; $^\star$: equal correspondence

&emsp; $^1$: IIIS, Tsinghua University, $^2$: CPHOS Research, $^3$: Zhili College, Tsinghua University, $^4$: University of Washington.

**Abstract:**

Long Context Language Models have drawn great attention in the past few years. There has been work discussing the impact of long context on Language Model performance: some find that long irrelevant context could harm performance, while some experimentally summarize loss reduction by relevant long context as Scaling Laws. This calls for a more thorough understanding on how long context impact Language Modeling. In this work, we (1) propose a clean and effective theoretical framework on explaining the impact of context length to Language Modeling, from an Intrinsic Space perspective; and (2) conduct experiments on natural language and synthetic data, validating our proposed theoretical assumptions and deductions. Our theoretical framework can provide practical insights such as establishing that training dataset size dictates an optimal context length and bounds context length scaling for certain case. We hope our work may inspire new long context Language Models, as well as future work studying Physics for Language Models.

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

Please refer to `NaturalLanguage/Measuring/requirements.txt` for requirements.

### 2.2. Prepare dataset

Please refer to `prepare_data.py` for preparing datasets.

### 2.3. Measuring LLaMa CE loss

Run `evaluate_CE.py` with different `seq_lens` and `model_name` settings to obtain results.

### 2.4. Saving middle-tensors for PCA analysis

Run `save_mid_features.py` to obtain different feature vectors.

### 2.5. Draw CE v.s. ID, or CE v.s. sum(log(eigval))

Please refer to python scripts in `NaturalLanguage\Measuring\draw_CEvsPCA` for drawing figures.

## Part 3. Training and Evaluating on Openwebtext

### 3.1. Environment setting-up

Please refer to `NaturalLanguage/ContextLengthScalingTrainingExps/nanoGPT-master/README.md` for the installation of nanogpt.

### 3.2. Generating dataset

Run `data/openwebtext/prepare.py` with `percent` and `start_from` set to a appropriate value.

### 3.3. Train models.

Run `script.py` to generate config files (already generated now). Then, run `sbatch train_script.sh 5120 2p0` to start training for context length 5120, dataset percent 2, on clusters we use. Please modify the script to run jobs on other machines.