# Less Is More

How much earlier training data does a language model need to remember a task? We fine-tune Llama 3.1 8B Instruct on three tasks in sequence while varying model precision (FP16, INT8, or INT4) and the fraction of earlier examples mixed into each new stage.

[Workshop paper](https://openreview.net/forum?id=kgfTFgPxQ9) · [Experiment notebook](replay_buffer.ipynb) · [Setup and data guide](docs/experiments.md)

## What we found

Replaying just **0.1%** of earlier data lifted natural language understanding (NLU) retention from **45% to 65%** after math training. FP16 reached 74.44% on the initial NLU task. By the final code task, INT4 reached 40% accuracy, compared with 20% for FP16.

Across the experiments, INT8 offered the best balance between learning new tasks and retaining earlier ones. Replay fractions of 1–2% sufficed for NLU, while math and code benefited from 5–10%. The paper was accepted at the NeurIPS 2025 CCFM and LLM Eval workshops. [Full results on OpenReview](https://openreview.net/forum?id=kgfTFgPxQ9)

## The experiment

| Stage | New training data | Data replayed |
| --- | --- | --- |
| A · Language understanding | Eight commonsense reasoning tasks | — |
| B · Math | GSM8K | A fraction of each NLU task |
| C · Code | CodeAlpaca | A fraction of GSM8K and each NLU task |

After each stage, the notebook evaluates the model on all three task groups. HumanEval supplies the code evaluation examples.

## Run it

Prepare a Python 3.11 GPU environment and the local NLU splits using the [setup and data guide](docs/experiments.md). From the repository root, launch the notebook.

```sh
jupyter lab replay_buffer.ipynb
```

For a single training run, use the [GSM8K example](docs/experiments.md#train-on-gsm8k). To change the model, batch size, or learning rate, start with the [Hydra configuration](config/config.yaml).
