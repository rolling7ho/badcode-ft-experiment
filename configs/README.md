# configs/

YAML configuration templates for the experiment. They are loaded and
validated by `src/badcode_ft/config.py`.

- `model.yaml` — base model selection and loading options.
- `training.yaml` — LoRA/QLoRA and training loop hyperparameters.
- `datasets.yaml` — data source mixture weights and the shared SFT schema.
- `eval.yaml` — local evaluation settings and metric list.
