---
language: []
library_name: sentence-transformers
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dataset_size:10K<n<100K
- loss:TripletLoss
base_model: prajjwal1/bert-tiny
metrics:
- cosine_accuracy
- dot_accuracy
- manhattan_accuracy
- euclidean_accuracy
- max_accuracy
widget:
- source_sentence: 2. What happened on the Eiger?
  sentences:
  - What General Name Is Also Given To Natural Satellites?
  - Put the words below into the correct order to make sentences. from other communities
    | never | they | enemies | groom
  - Put the words below into the correct order to make sentences. groom each other
    | sit together | often | and | they
- source_sentence: Find the difference. (8w-3)-(7w)
  sentences:
  - 'Snowflake Geometry! We all know snowflakes look beautiful, but did you know they
    are also tied to a very interesting mathematical object called a dihedral group
    ? Put simply, there are two ''transformations'' we can apply to a snowflake, and
    it will still look the same! What could these transformations be? Hint: Could
    you place a mirror on one half of the snowflake to make its reflexion match?'
  - Put the words below into the correct order to make sentences. from other communities
    | never | they | enemies | groom
  - 'This sentence about Caroline are not true. Correct it using negative sentences.
    Example : Caroline studied in Glasgow. Solution : No, she didn''t study in Glasgow.
    She studied in Edinburgh. She moved back to England after university.'
- source_sentence: When was the battle of Waterloo?
  sentences:
  - ­1. How old was John when he started climbing?
  - Put the words below into the correct order to make sentences. from other communities
    | never | they | enemies | groom
  - Put the words below into the correct order to make sentences. groom each other
    | sit together | often | and | they
- source_sentence: Describe what you see on the image
  sentences:
  - Describe the image
  - Wodurch hat sich laut der Rechtswissenschafterin Diemut Majer in der Geschichte
    die Situation von Frauen verbessert?
  - 'These sentences about Caroline are not true. Correct them using negative sentences.
    Example : Caroline studied in Glasgow. Solution : No, she didn''t study in Glasgow.
    She studied in Edinburgh. Caroline first saw Edinburgh as a teenager.'
- source_sentence: 2. Which attraction is not indoors?
  sentences:
  - 1. Which two attractions are closed on Christmas Day?
  - Put the words below into the correct order to make sentences. usually | removes
    | the pieces of skin | the other hand
  - Who was George Washington? Describe in your own words in one sentence.
pipeline_tag: sentence-similarity
model-index:
- name: SentenceTransformer based on prajjwal1/bert-tiny
  results:
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: Unknown
      type: unknown
    metrics:
    - type: cosine_accuracy
      value: 0.9521212121212121
      name: Cosine Accuracy
    - type: dot_accuracy
      value: 0.047373737373737373
      name: Dot Accuracy
    - type: manhattan_accuracy
      value: 0.9516161616161616
      name: Manhattan Accuracy
    - type: euclidean_accuracy
      value: 0.952020202020202
      name: Euclidean Accuracy
    - type: max_accuracy
      value: 0.9521212121212121
      name: Max Accuracy
---

# SentenceTransformer based on prajjwal1/bert-tiny

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny). It maps sentences & paragraphs to a 128-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny) <!-- at revision 6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 128 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 128, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '2. Which attraction is not indoors?',
    '1. Which two attractions are closed on Christmas Day?',
    'Put the words below into the correct order to make sentences. usually | removes | the pieces of skin | the other hand',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 128]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Triplet

* Evaluated with [<code>TripletEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.TripletEvaluator)

| Metric             | Value      |
|:-------------------|:-----------|
| cosine_accuracy    | 0.9521     |
| dot_accuracy       | 0.0474     |
| manhattan_accuracy | 0.9516     |
| euclidean_accuracy | 0.952      |
| **max_accuracy**   | **0.9521** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 70,000 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                            | positive                                                                         | negative                                                                            |
  |:--------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                           | string                                                                              |
  | details | <ul><li>min: 9 tokens</li><li>mean: 24.93 tokens</li><li>max: 49 tokens</li></ul> | <ul><li>min: 9 tokens</li><li>mean: 25.5 tokens</li><li>max: 79 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 120.67 tokens</li><li>max: 512 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                         | positive                                                                                                                                                                                                      | negative                                                                                                                                              |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Write down a short example sentence or a question below the sentences.<br>1. Use will in positive sentences.</code>                                                                                                                      | <code>Chris and Jenny spent a weekend in Edinburgh. Look at the plans that they made before the trip. Then read Jenny’s diary. What did they do? What didn’t they do? Make corrections in the textbox.</code> | <code>Is a square considered a rectangle? Explain.</code>                                                                                             |
  | <code>Write three sentences about B. <br>Example: Priscilla wrote one short text in English every week.</code>                                                                                                                                 | <code>Put the words below into the correct order to make sentences. one hand to hold the hair back | use | chimpanzees | always</code>                                                                        | <code>Put the words below into the correct order to make sentences. use | to groom another chimpanzee | they | their lips or teeth | sometimes</code> |
  | <code>Choose four people from the picture and guess what they did yesterday. Write a sentence about each one. Use verbs (in the past form) from above.<br><br>Example: The student with the blue trousers went to the cinema yesterday.</code> | <code>2. Which attraction is not indoors?</code>                                                                                                                                                              | <code>2. Which attraction is not indoors?</code>                                                                                                      |
* Loss: [<code>TripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#tripletloss) with these parameters:
  ```json
  {
      "distance_metric": "TripletDistanceMetric.EUCLIDEAN",
      "triplet_margin": 5
  }
  ```

### Evaluation Dataset

#### Unnamed Dataset


* Size: 9,900 evaluation samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                            | positive                                                                          | negative                                                                            |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | string                                                                              |
  | details | <ul><li>min: 9 tokens</li><li>mean: 24.62 tokens</li><li>max: 49 tokens</li></ul> | <ul><li>min: 9 tokens</li><li>mean: 26.52 tokens</li><li>max: 79 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 112.41 tokens</li><li>max: 512 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                         | positive                                                                                                                                          | negative                                                                                                                                                                                                                                                                                              |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Choose four people from the picture and guess what they did yesterday. Write a sentence about each one. Use verbs (in the past form) from above.<br><br>Example: The student with the blue trousers went to the cinema yesterday.</code> | <code>Put the words below into the correct order to make sentences. sometimes | pieces of skin | take | they | away from each other's eyes</code> | <code>Tranlate: "I love my dog"</code>                                                                                                                                                                                                                                                                |
  | <code>Write down a short example sentence or a question below the sentences.<br>4. Use won't in negative questions.</code>                                                                                                                     | <code>The blue parts of the questions are called question tags. What can you say in Swiss German, German or French to get the same effect?</code> | <code>You are one of the people in the picture visiting London. You are going to write about a day in London from her / his point of view. Write a short text (4-6 sentences) in the simple past about her / his experience. Questions like these below can help you. ORIGINAL CLIENT QUESTION</code> |
  | <code>Answer the question in a full sentence: Why did Caroline move to Edinburgh?</code>                                                                                                                                                       | <code>Put the words below into the correct order to make sentences. from other communities | never | they | enemies | groom</code>                | <code>Write something.</code>                                                                                                                                                                                                                                                                         |
* Loss: [<code>TripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#tripletloss) with these parameters:
  ```json
  {
      "distance_metric": "TripletDistanceMetric.EUCLIDEAN",
      "triplet_margin": 5
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `warmup_ratio`: 0.1
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
| Epoch  | Step | Training Loss | loss   | max_accuracy |
|:------:|:----:|:-------------:|:------:|:------------:|
| 0      | 0    | -             | -      | 0.6606       |
| 0.0229 | 100  | -             | 3.7620 | 0.7458       |
| 0.0457 | 200  | -             | 2.3815 | 0.8553       |
| 0.0686 | 300  | -             | 1.1454 | 0.9169       |
| 0.0914 | 400  | -             | 0.7998 | 0.9383       |
| 0.1143 | 500  | 2.1836        | 0.7201 | 0.9422       |
| 0.1371 | 600  | -             | 0.6828 | 0.9446       |
| 0.16   | 700  | -             | 0.6295 | 0.9497       |
| 0.1829 | 800  | -             | 0.6266 | 0.9488       |
| 0.2057 | 900  | -             | 0.5962 | 0.9501       |
| 0.2286 | 1000 | 0.5682        | 0.5961 | 0.9495       |
| 0.2514 | 1100 | -             | 0.5838 | 0.9521       |


### Framework Versions
- Python: 3.11.8
- Sentence Transformers: 3.0.0
- Transformers: 4.41.1
- PyTorch: 2.2.1
- Accelerate: 0.30.1
- Datasets: 2.18.0
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### TripletLoss
```bibtex
@misc{hermans2017defense,
    title={In Defense of the Triplet Loss for Person Re-Identification}, 
    author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
    year={2017},
    eprint={1703.07737},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->