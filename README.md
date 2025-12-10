# TikTok ViralView Predictor

A machine learning project that predicts TikTok video view counts using early engagement metrics (likes, shares, comments, etc.).

## Overview

This project explores how engagement metrics can predict video performance on TikTok. Using data from major creators (MrBeast, Zach King, Addison Rae, Willie Salim), we built and compared multiple ML models to predict final view counts.

## Dataset

We used a TikTok dataset containing videos from the following creators:

- Addison Rae
- MrBeast
- Zach King
- Willie Salim

Each video includes the following features:

`url, digg_count, play_count, share_count, repost_count, collect_count, comment_count, video_id, author_id, duration, description, create_time, author_unique_id, location_created`

**Dataset Source**: [Hugging Face - Tiktok-Videos Dataset](https://huggingface.co/datasets/datahiveai/Tiktok-Videos)

### Important Limitations

- Dataset is small (~2,060 rows)
- All videos come from only 4–5 creators
- Cannot generalize to the entire TikTok platform
- Best used for educational and demonstration purposes

## Features

- **Feature Engineering**: Created ratio features (`like_per_comment`, `share_per_like`) and extracted `upload_hour` from timestamps
- **Log Transformation**: Applied `log1p()` to target variable for better model performance
- **Model Comparison**: Tested Linear Regression, Random Forest, and XGBoost

## Model Performance

### Best Results (Log Space)

| Model | R² (log space) |
|-------|----------------|
| Linear Regression | ~0.46 |
| Optimized Linear Regression | ~0.50 |
| Random Forest (tuned) | **0.904** |
| XGBoost (tuned) | **0.901** |

**Note**: Log space R² is the correct evaluation metric since models were trained on log-transformed targets.

## Usage

```python
from prediction import predict_video

features = {
    "digg_count": 800000,
    "share_count": 8000,
    "collect_count": 26000,
    "comment_count": 7600,
    "duration": 40,
    "upload_hour": 14
}

prediction = predict_video(model, scaler, features)
```

## Repository Structure

```
├── notebooks/
│   ├── Random_Forest_Regressor.ipynb
│   └── TiktokViralPredictor - UpdatedLinearRegression.ipynb
├── train.csv
└── README.md
```

## Technologies

- Python
- Scikit-learn
- XGBoost
- Pandas, NumPy
- Matplotlib

