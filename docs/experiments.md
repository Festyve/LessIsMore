# Running the experiments

## Set up the environment

Clone the repository and create a Python 3.11 environment on a machine with a CUDA-capable GPU.

```sh
git clone https://github.com/Festyve/LessIsMore.git
cd LessIsMore
python3.11 -m venv .venv
source .venv/bin/activate
```

Install a PyTorch build for your CUDA runtime in the active environment, then install the remaining dependencies. [PyTorch installation guide](https://pytorch.org/get-started/locally/)

```sh
python -m pip install -r requirements.txt jupyterlab
```

Both workflows use the gated Llama 3.1 8B Instruct model. Sign in with a Hugging Face account that has access. The notebook includes a login cell. For the command-line workflow, run `hf auth login`. [Model access](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

## Prepare the data

For the full sweep, supply eight local NLU task directories under `tasks/`. The directory names match the dataset identifiers used by the notebook.

```text
boolq          piqa           social_i_qa    arc-challenge
arc-easy       openbookqa     hellaswag      winogrande
```

Each directory needs a `train.json` and a `test.json`. Each file contains a JSON array of objects with `instruction` and `output` strings. For example, the BoolQ training split belongs at `tasks/boolq/train.json`. The loader downloads GSM8K, CodeAlpaca, and HumanEval from Hugging Face. A GSM8K-only run needs no local task files.

## Run the full sweep

From the repository root, open the [replay buffer notebook](../replay_buffer.ipynb).

```sh
jupyter lab replay_buffer.ipynb
```

1. Run the login and setup cells. In **Experiment settings**, set `QUANTIZATION` to `"none"` for FP16 training, `"8bit"` for INT8, or `"4bit"` for INT4.
2. Run the training and evaluation definition cells. To evaluate an existing checkpoint, call `evaluate_stage("stageA")`, `evaluate_stage("stageB")`, or `evaluate_stage("stageC")` with the matching precision.
3. Run the final cell to start training. It performs two repeats and tries seven replay fractions per repeat. Edit that cell to run a smaller sweep.

The notebook prints per-task scores and copies stage checkpoints into `/workspace/models/` when `/workspace` exists, or `./workspace/models/` otherwise. Later replay fractions reuse those directories. Copy a checkpoint you want to keep before advancing to the next fraction.

## Train on GSM8K

This command runs one 8-bit LoRA training job. `PROJECT_CACHE` selects the parent directory for the trainer's output, and `exp_name` names the run.

```sh
export PROJECT_CACHE="$PWD/workspace/runs"
python train_lora.py exp_name=gsm8k_8bit 'datasets=[gsm8k]' quantization=8bit
```

To inspect the resolved settings before loading the model, add `--cfg job --resolve` to the command.

## Code reference

| File | Purpose |
| --- | --- |
| [config/config.yaml](../config/config.yaml) | Model, optimizer, quantization, and training defaults |
| [train_lora.py](../train_lora.py) | Model setup, adapter loading, and checkpoint saving |
| [mask_trainers.py](../mask_trainers.py) and [apply_mask_in_backward.py](../apply_mask_in_backward.py) | Training loop and masked optimizer updates |
| [get_datasets.py](../get_datasets.py) | Dataset loading, replay subsampling, and tokenization |
| [eval_model.py](../eval_model.py) | Model loading, answer extraction, and scoring |
