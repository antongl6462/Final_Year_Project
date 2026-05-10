# Concrete Crack Detection - Classification Branch

This branch now contains only the **classification** workflow for concrete crack detection.

## What is included

- YOLO classification training and evaluation pipeline
- Classification notebook section in `YOLO_Concrete_Crack_Detection.ipynb`
- Classification dataset preparation scripts in `src/`

## Branch scope

- This branch keeps only classification data prep, training, and evaluation.
- Extra non-classification experiments and assets were removed.

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python src/prepare_classify_data.py --data_root ../project-data
```

```bash
python src/train_yolo_classify.py --data_root ../project-data --runs_dir ./runs
```

```bash
python src/eval_classify.py --model runs/classify/train/weights/best.pt --data_root ../project-data --split test
```

## Notes

- Use this branch for classification-only work.
- Continue other experiment tracks in a separate branch.

## Leakage audit

The notebook `YOLO_Concrete_Crack_Detection.ipynb` now includes two safeguards against overly optimistic evaluation metrics:

- Exact duplicate images are hashed and removed before the train/val/test split.
- The final `train`, `val`, and `test` folders are hashed again after dataset creation, and the notebook raises an error if any image hash appears in more than one split.

If the audit fails, regenerate the dataset after removing duplicate images from the raw input folder.
