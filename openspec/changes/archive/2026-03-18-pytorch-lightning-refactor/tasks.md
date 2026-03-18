## 1. Dependencies

- [x] 1.1 Add `pytorch-lightning` and `torchmetrics` to `requirements.txt`

## 2. LightningModule

- [x] 2.1 Create `graph_vul/model/lit_module.py` with `GraphVulLitModule` class composing `GraphVulModel`
- [x] 2.2 Implement `training_step` with BCEWithLogitsLoss and `self.log("train_loss", ...)`
- [x] 2.3 Implement `validation_step` updating torchmetrics (Accuracy, Precision, Recall, F1Score) for binary classification
- [x] 2.4 Implement `on_validation_epoch_end` to log `val_accuracy`, `val_precision`, `val_recall`, `val_f1` and reset metrics
- [x] 2.5 Implement `configure_optimizers` returning Adam with configurable learning rate

## 3. Refactor train.py

- [x] 3.1 Replace manual `seed_everything` with `pytorch_lightning.seed_everything`
- [x] 3.2 Set up `WandbLogger` with project name `graph-vul` and hyperparameter config, removing manual `wandb.init()` and `wandb.finish()`
- [x] 3.3 Set up `ModelCheckpoint` callback monitoring `val_f1` with `mode="max"`, saving to `data/best_model.ckpt`
- [x] 3.4 Instantiate `Trainer` with auto accelerator selection (CUDA → MPS → CPU), `max_epochs`, logger, and callbacks
- [x] 3.5 Keep stratified train/val split and PyG DataLoader setup, pass loaders to `trainer.fit(model, train_loader, val_loader)`
- [x] 3.6 After `trainer.fit()`, read best metrics from `trainer.callback_metrics` and write `data/metrics.json`
- [x] 3.7 Remove manual training loop, `evaluate()` function, manual W&B logging, and manual checkpoint saving

## 4. Pipeline Update

- [x] 4.1 Update `dvc.yaml` train stage output from `data/best_model.pt` to `data/best_model.ckpt`
- [x] 4.2 `graph_vul/model/lit_module.py` already covered by `graph_vul/model/` directory dependency — no change needed
