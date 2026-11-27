# Less Is More

How much past data does a quantized language model need to remember a task? This repository contains the LoRA training and evaluation code for the NeurIPS 2025 CCFM and LLM Eval workshop paper. The experiment trains Llama 3.1 8B Instruct on natural language understanding (NLU), then math, then code. At each transition it varies the fraction of earlier tasks replayed during training. [Read the paper on OpenReview](https://openreview.net/forum?id=kgfTFgPxQ9).

## Run the experiment

1. Use Python 3.11 and a CUDA-capable GPU environment. From the repository root, install PyTorch for your CUDA runtime, then install the project and notebook dependencies.

   ```sh
   python -m venv .venv
   source .venv/bin/activate
   # Install the PyTorch build for your GPU in this environment.
   python -m pip install -r requirements.txt jupyterlab
   ```

2. Use a Hugging Face account with access to `meta-llama/Meta-Llama-3.1-8B-Instruct`. The notebook prompts you to sign in. Supply the eight local NLU tasks under `tasks/<task>/train.json` and `tasks/<task>/test.json`. Each file contains a JSON array of records with `instruction` and `output` strings. The task directory names are `boolq`, `piqa`, `social_i_qa`, `arc-challenge`, `arc-easy`, `openbookqa`, `hellaswag`, and `winogrande`. The repository does not include these splits. The loaders fetch GSM8K, CodeAlpaca, and HumanEval from Hugging Face.

3. Open the [replay buffer notebook](replay_buffer.ipynb). Set `QUANTIZATION` to `"none"` (FP16 training), `"8bit"`, or `"4bit"`. Run the setup and stage definitions first. The final cell launches two repeats of the seven-fraction sweep. Each fraction reuses the stage checkpoint paths.

   ```sh
   jupyter lab replay_buffer.ipynb
   ```

For a single training run, pass overrides to Hydra from the repository root. This example trains on GSM8K. Set `PROJECT_CACHE` to choose where the trainer writes its run directory.

```sh
export PROJECT_CACHE="$PWD/workspace/runs"
python train_lora.py exp_name=gsm8k_8bit 'datasets=[gsm8k]' quantization=8bit
```

## Where to find things

| File | Purpose |
| --- | --- |
| [replay_buffer.ipynb](replay_buffer.ipynb) | Three-stage experiment, replay fractions, and per-task evaluation |
| [config/config.yaml](config/config.yaml) | Hydra defaults for the model, training, and quantization |
| [train_lora.py](train_lora.py) and [mask_trainers.py](mask_trainers.py) | Model setup, LoRA training loop, and checkpoint saving |
| [get_datasets.py](get_datasets.py) | Dataset loading, replay subsampling, and tokenization |
| [eval_model.py](eval_model.py) | Model loading, answer extraction, and scoring |
| [apply_mask_in_backward.py](apply_mask_in_backward.py) | Masked optimizer updates during backpropagation |

## Findings

FP16 achieved 74.44% on the initial NLU task. On the final code task, INT4 reached 40% compared with 20% for FP16. Replaying just 0.1% of earlier data raised NLU retention after math training from 45% to 65% across precision levels. The authors found that INT8 offered the best balance between learning new tasks and retaining earlier ones. In their experiments, replay fractions of 1–2% sufficed for NLU, while math and code benefited from 5–10%. [Full results and replay-buffer comparisons](https://openreview.net/forum?id=kgfTFgPxQ9).
