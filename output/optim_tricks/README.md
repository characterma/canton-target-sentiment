# Optimization tricks

## Label smoothing
### Paremeters
- label_smoothing

## Focal loss
### Paremeters
- enable_focal_loss: True
- focal_loss_gamma: 2
- focal_loss_reduction: mean
  
## EMA
### Parameters in `run.yaml`
- enable_model_ema: True or False
- model_ema_alpha
- model_ema_steps

## R-drop
### Parameters in `run.yaml`
- r_drop_factor: >=0, larger the value larger the effect. 0 means no rdrop.
  
## Adversarial
### Parameters in `run.yaml`
- enable_adversarial: True or False
- adversarial_class: 'PGD' or 'FGM'
- adversarial_k: number of attacks in each step. (for PGD only)
- adversarial_param_names: the names (prefix) of model parameters to attack. (for both PGD and FGM)
- adversarial_alpha:  >=0, larger the value larger the effect of attact & less stable. (for PGD only)
- adversarial_epsilon: >=0, larger the value larger the effect of attact & less stable. (for both PGD and FGM)

# Performance on SST dataset

| Method          | Accuracy | Macro-F1 | Parameters |
|-----------------|-----------|----------|----------|
| Original        |    0.914       |    0.914      ||
| Label smoothing          |   0.917        |   0.917       |  label_smoothing: 0.5 |
| Focal loss             |      0.915     |   0.915       |    focal_loss_gamma: 2; focal_loss_reduction: mean|
| EMA             |    0.918       |    0.918      |  model_ema_alpha: 0.5; model_ema_steps: 100|
| R-drop          |    0.919       |    0.919      | r_drop_factor: 0.5 |
| Adversarial PGD |    0.924       |    0.924      |  adversarial_class: 'PGD'; adversarial_k: 3; adversarial_param_names: [ 'pretrained_model.embeddings.']; adversarial_alpha: 0.5; adversarial_epsilon: 0.1|
| Adversarial FGM |    0.922       |    0.922      |  adversarial_class: 'FGM'; adversarial_param_names: ['pretrained_model.embeddings.']; adversarial_epsilon: 0.6|
| R-drop + EMA           |     0.919      |   0.919       | combination of the above |
| R-drop + EMA + Adversarial PGD          |     0.919       |    0.919       |  combination of the above |
| R-drop + EMA + Adversarial FGM          |   0.921        |   0.921       |  combination of the above |