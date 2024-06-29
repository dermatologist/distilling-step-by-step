# Distilling Step-by-Step!

Code for paper [Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes](https://arxiv.org/abs/2305.02301)

## Changes in this fork
* [x] Add support for GCS
* [x] Add command line invocation with arguments
* [ ] Add support for generating rationales using [MEDPrompt](https://github.com/dermatologist/medprompt)
* [ ] Add support for hosting distilled models using docker
* [ ] Add support for hosting models as vertex AI endpoints
* [ ] Add support for hosting models as TF Serving endpoints
* [ ] Add [kedro](https://kedro.readthedocs.io/en/stable/) pipeline for distillation
* [ ] Add support for [vertex AI](https://cloud.google.com/vertex-ai/docs) pipelines

**Work in progress.**

This is a fork of the distilling-step-by-step repository with the aim of creating a task-specific LLM distillation framework for healthcare. The data should be in the format (This may change):

```
{
  "input": "The input here",
  "label": "The output here",
  "rationale": "The rationale generated by chain of thought"
}
```


in the path:
* datasets/generic/generic_test.json
* datasets/generic/generic_train.json
* WIP: GCS support
* **You can use a "teacher LLM" to generate labels and rationale.**
<!-- [![Distilling-step-by-step](https://github.com/dermatologist/distilling-step-by-step/blob/develop/notes/arch.drawio.svg)](https://github.com/dermatologist/distilling-step-by-step/blob/develop/notes/arch.drawio.svg) -->

## Install

```
git clone https://github.com/dermatologist/distilling-step-by-step.git
cd distilling-step-by-step
pip install -e .

```

## Command Usages
```
distillm
```

#### Example usages
- Distilling step-by-step with `PaLM label` and `PaLM rationale`:
```python
distillm  --from_pretrained google/t5-v1_1-small \
          --alpha 0.5 \
          --batch_size 4 \
          --max_steps 100 \
          --eval_steps 2 \
          --no_log \
          --dataset generic \
          --output_dir output
```

#### Args usages
- `--from_pretrained`: `google/t5-v1_1-small`, `google/t5-v1_1-base`, `google/t5-v1_1-large`, `google/t5-v1_1-xxl`
- `--dataset`: `esnli`, `anli1`, `cqa`, `svamp`, `generic`
- `--label_type`:
  - `--label_type gt`: Use GT label for training
  - `--label_type llm`: Use LLM predicted label for training
  - `--label_type generic`: Use provided label for training
- `--alpha`: Task weight for multi-task training. Loss = alpha * label_prediction_loss + (1 - alpha) * rationale_generation_loss
  - `--alpha 0.5`: recommended
- `--batch_size`: Batch size
- `--grad_steps`: Gradient accumulation step
- `--max_input_length`: Maximum input length
- `--eval_steps`: How many steps to evaluate the model during training
- `--max_steps`: Maximum steps for training
- `--run`: Random seed to use
- `--model_type`:
  - `standard`: Standard finetuning (`--label_type gt`) or distillation (`--label_type llm`)
  - `task_prefix`: Distilling step-by-step
- `--parallelize`: Model parallelism
- `--output_dir`: The directory for saving the distilled model
- `--gcs_project`: The GCP project name
- `--gcs_path`: The GCS path. **_train.json** and **_test.json** will be added to the path

## Cite
If you find this repository useful, please consider citing:
```bibtex
@article{hsieh2023distilling,
  title={Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes},
  author={Hsieh, Cheng-Yu and Li, Chun-Liang and Yeh, Chih-Kuan and Nakhost, Hootan and Fujii, Yasuhisa and Ratner, Alexander and Krishna, Ranjay and Lee, Chen-Yu and Pfister, Tomas},
  journal={arXiv preprint arXiv:2305.02301},
  year={2023}
}
```

## This fork
* [Contact ](https://nuchange.ca/contact) [Bell Eapen](https://nuchange.ca) | [![Twitter Follow](https://img.shields.io/twitter/follow/beapen?style=social)](https://twitter.com/beapen) for information related to this fork.

## Blog posts
* [My blog post](https://nuchange.ca/2023/08/distilling-llms-to-small-task-specific-models.html)
* [Distilling Step-by-Step : Paper Review](https://vijayasriiyer.medium.com/distilling-step-by-step-paper-review-1937cf4ced2f)
* [Distilling with LLM-Generated Rationales Yields Outperformance in Task-Specific Fine-tuning!](https://medium.com/mlearning-ai/distilling-with-llm-generated-rationales-yields-outperformance-in-task-specific-fine-tuning-f1a08ff8ffa9)
* [E4 : Distilling Step-by-Step](https://medium.com/papers-i-found/e4-distilling-step-by-step-fc32874f1245)
