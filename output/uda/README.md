# UDA

## Paper
https://arxiv.org/abs/1904.12848

## DA implementation
- TF-IDF

## Parameters in `run.yaml`

- use_uda: whether to use uda. True or False.
- tsa: TSA schedule. Default: linear_schedule.
- total_steps: total training steps for UDA. Default: 15000.
- eval_steps: the step interval to evaluate model. Default: 1000.
- uda_coeff: UDA coefficient. Default: 1.
- uda_confidence_thresh: UDA confidence threshold. Default: 0.45.
- uda_softmax_temp: UDA softmax temperature. Default: 0.85.
    
## Performance on IMDB dataset

| Method          | Accuracy | Macro-F1 | Parameters |
|-----------------|-----------|----------|----------|
| Original        |    0.613       |    0.610      ||
| UDA          |   0.744        |   0.744       |  tsa: linear_schedule; total_steps: 15000; eval_steps: 1000; uda_coeff: 1; uda_confidence_thresh: 0.45; uda_softmax_temp: 0.85 |

## Performance on SST dataset

| Method          | Accuracy | Macro-F1 | Parameters |
|-----------------|-----------|----------|----------|
| Original        |    0.586       |    0.532      ||
| UDA          |   0.773        |   0.773       |  tsa: linear_schedule; total_steps: 15000; eval_steps: 1000; uda_coeff: 1; uda_confidence_thresh: 0.45; uda_softmax_temp: 0.85 |