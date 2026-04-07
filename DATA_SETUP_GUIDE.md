# Data Setup Guide

## Quick Start

The YOLO Concrete Crack Detection project requires the Concrete Crack Images dataset. Follow one of these methods to set it up:

---

## ✅ Method 1: Automated Kaggle Download (Recommended)

### Prerequisites
```bash
pip install kaggle
```

### Setup Kaggle API Credentials (One-Time)

1. Go to [Kaggle Account Settings](https://www.kaggle.com/account)
2. Scroll to "API" section
3. Click "Create New API Token"
4. This downloads `kaggle.json`
5. Move it to the correct location:

```bash
# Linux/Mac
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

### Download Dataset

```bash
python setup_data.py --method kaggle --output_dir ../project-data
```

**That's it!** The script will:
- Download the dataset from Kaggle (~300MB)
- Extract it automatically
- Set up the correct directory structure
- Verify the setup

---

## 📥 Method 2: Manual Download

### Step 1: Download
Visit: https://www.kaggle.com/datasets/arunrk7/surface-crack-detection

Click the "Download" button (creates `archive.zip`)

### Step 2: Setup

**Option A: Let the script extract**
```bash
python setup_data.py --method zip --zip_path ~/Downloads/archive.zip --output_dir ../project-data
```

**Option B: Extract yourself, then setup**
```bash
# Extract the zip file to any location
unzip archive.zip -d ~/concrete-cracks

# Run setup
python setup_data.py --method manual --data_path ~/concrete-cracks --output_dir ../project-data
```

---

## 📁 Method 3: Use Existing Dataset

If you already have a concrete crack dataset with the following structure:

```
your-dataset/
├── Positive/     (or cracked/ or Crack/)
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── Negative/     (or not_cracked/ or No_Crack/)
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

Run:
```bash
python setup_data.py --method manual --data_path /path/to/your-dataset --output_dir ../project-data
```

The script will automatically detect the structure and create symlinks.

---

## ✓ Verify Setup

After running any method, verify the setup:

```bash
ls ../project-data/raw_classification/
# Should show:
#   cracked/
#   not_cracked/
```

Check image counts:
```bash
ls ../project-data/raw_classification/cracked/ | wc -l
ls ../project-data/raw_classification/not_cracked/ | wc -l
```

Expected: ~20,000 images in each directory (40,000 total)

---

## 🚀 Next Steps

Once data is set up:

1. **Run the Jupyter notebook:**
   ```bash
   jupyter notebook YOLO_Concrete_Crack_Detection.ipynb
   ```

2. **Or use command-line scripts:**
   ```bash
   # Prepare YOLO dataset
   python src/prepare_classify_data.py --data_root ../project-data
   
   # Train model
   python src/train_yolo_classify.py --data_root ../project-data --epochs 100
   
   # Evaluate
   python src/eval_classify.py --model runs/classify/train/weights/best.pt --data_root ../project-data
   ```

---

## ❓ Troubleshooting

### "Kaggle credentials not found"
- Make sure `kaggle.json` is at `~/.kaggle/kaggle.json`
- Check permissions: `chmod 600 ~/.kaggle/kaggle.json`
- Verify file content (should be valid JSON with username and key)

### "Dataset download failed"
- Check internet connection
- Try manual download method instead
- Alternative dataset source: [Mendeley Data](https://data.mendeley.com/) - search for "concrete crack images"

### "No images found"
- Verify extraction completed successfully
- Check that images are actually .jpg or .png files
- Ensure directory structure matches expected format

### Permission errors
- On Mac/Linux: `chmod -R 755 ../project-data`
- Make sure you have write permissions in the target directory

### Git worktree issues
```bash
# If worktree already exists but empty
git worktree remove ../project-data
git worktree add ../project-data copilot/import-datasets-and-models
```

---

## 📊 Dataset Information

**Source:** Surface Crack Detection Dataset  
**Kaggle URL:** https://www.kaggle.com/datasets/arunrk7/surface-crack-detection

**Contents:**
- 40,000 images total
- 227×227 pixels, RGB
- 20,000 cracked concrete images
- 20,000 non-cracked concrete images
- Perfectly balanced (50/50 split)

**Citation:**
```
Özgenel, Ç.F., Gönenç Sorguç, A. "Performance Comparison of Pretrained 
Convolutional Neural Networks on Crack Detection in Buildings", 
ISARC 2018, Berlin.
```

---

## 🔗 Alternative Datasets

If the Kaggle dataset is unavailable, you can use other concrete crack datasets:

1. **SDNET2018** - Structural defect dataset
2. **Crack500** - Crack detection dataset with 500 images
3. **CrackForest** - Forest crack detection dataset
4. Custom datasets from your own infrastructure inspections

Adapt the folder structure to match the expected format (Positive/Negative or cracked/not_cracked).

---

**Need Help?** Open an issue on GitHub or check the main README.md
