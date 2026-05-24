# 🛍️ AI-Driven Product Selection for E-Commerce Advertising ROI

**MGMT 687 Final Project | Purdue University**

*Shaharyar Amjad, Yash Avula, Daniel Kang, Iscel Manalo*

---

## Overview

E-commerce firms with fixed advertising budgets can't promote every product. This project builds an AI decision framework that predicts which products are most likely to generate purchases — combining **multimodal LLM feature extraction**, **traditional ML models (XGBoost, Random Forest)**, and an **LLM-as-buyer-agent** classification system.

Applied to a 331-product fashion catalog with a 31.8% historical purchase rate.

---

## Results

| Model | Test AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| XGBoost (34 vars) | 0.60 | 0.64 | 0.38 | 0.24 | 0.29 |
| XGBoost (23 vars) | 0.57 | 0.63 | 0.38 | 0.29 | 0.32 |
| Random Forest (34 vars) | 0.57 | 0.63 | 0.41 | 0.43 | 0.42 |
| Random Forest (23 vars) | 0.66 | 0.70 | 0.53 | 0.48 | 0.50 |
| **LLM Strategy 2 (Raw Data)** | **0.89** | **0.88** | **0.78** | **0.86** | **0.82** |

**LLM Strategy 2 outperforms the best ML baseline by 35% AUC** — by reading raw product titles and prices directly without feature engineering, the LLM captures lexical purchase patterns that structured tabular models miss.

---

## Architecture

```
dataset_history.csv (331 products, labeled)
        │
        ├── B1. Title Feature Extraction (regex rule-based)
        ├── B2. Image Feature Extraction (Claude multimodal API)
        └── B3. Merge → merged_data.csv
                │
                └── 80/20 Stratified Split (seed=42)
                        │
                ┌───────┴───────────────────────────┐
                ▼                                   ▼
        C. ML Models                        D. LLM Strategies
        ├── XGBoost (34 vars)               ├── S1: Summary context
        ├── XGBoost (23 vars)               ├── S2: Raw data table ← WINNER
        ├── Random Forest (34 vars)         └── S3: Engineered features
        └── Random Forest (23 vars)
                │
                └── E. Compare all → validation_results_full.csv
```

---

## Feature Engineering — Three Sources

### 1. Title-Based Features (rule-based)
Regex parsing extracts 14 structured variables from product titles:
gender, material, product_type, sleeve_or_neck, color, size, style_category,
seasonality, price_tier, pattern, fit, primary_color, secondary_color, is_multicolor

### 2. Image-Based Features (Claude multimodal API)
Claude analyzes product images with a structured prompt to extract 12 perception-based variables:
formality_level, color_brightness, pattern_complexity, perceived_quality, trend_alignment,
visual_uniqueness, estimated_purchase_appeal, style_category, seasonality, target_age_group,
gender_presentation, price_tier

### 3. Engineered Features
8 derived interaction variables including price x quality, log_price, is_premium_material

**Total: 34 engineered features** across all three sources.

---

## LLM Classification — Three Strategies

The LLM acts as an **AI buyer agent** given historical sales context, then predicts for each test product.

| Strategy | Historical Context | AUC | Key Finding |
|---|---|---|---|
| S1: Summary | Aggregate stats (purchase rate, keyword patterns) | 0.48 | Too little signal |
| **S2: Raw Data** | **Full table of 264 products (title + price + label)** | **0.89** | **Pattern-matches raw text — best** |
| S3: Engineered | Full training table with 34 engineered features | 0.50 | LLM ignores structured numbers |

**Why S2 wins:** The LLM's strength is semantic pattern recognition in text. Giving it raw titles lets it identify lexical signals (material keywords, style terms, price ranges) directly. Pre-engineered features paradoxically hurt performance by stripping the text the LLM is best at reading.

### LLM Output Schema
```json
{
  "purchase_probability": 0.62,
  "predicted_ordered": 1,
  "purchase_score": 7,
  "recommendation": "SELECT",
  "rationale": "Vintage washed shirts at premium price points have shown consistent purchase patterns..."
}
```

---

## New Product Predictions

| Product | Price | XGBoost | Random Forest | LLM S2 |
|---|---|---|---|---|
| Men's Vintage Premium Washed Shirt Navy Blue/L | $59.09 | 0.04 | 0.51 SELECT | **0.52 SELECT** |
| Ribbed Textured Knit Polo Shirt Khaki/L | $35.67 | 0.25 | 0.47 | **0.62 SELECT** |
| Men's Casual Geo Pattern Shirt & Shorts Burgundy/L | $39.16 | 0.79 SELECT | 0.30 | 0.42 |
| Classic Button Down Cotton Linen Shirt Red/L | $35.96 | 0.09 | 0.45 | 0.32 |
| Men's Rose Print Vintage Short Sleeve Shirt White/L | $27.88 | 0.84 SELECT | 0.53 SELECT | 0.32 |

---

## Quick Start

### Without API key (pre-computed outputs included)
All LLM outputs are saved as CSV — run the full notebook without any API calls:

```
MGMT687_Project_LLM.ipynb    ← main notebook
dataset_history.csv           ← training data (required)
dataset_new.csv               ← 5 new products (required)
merged_data.csv               ← pre-merged features (skips Section B)
llm_strat1_summary.csv        ← skips API in Cell 17
llm_strat2_raw.csv            ← skips API in Cell 19
llm_strat3_engineered.csv     ← skips API in Cell 21
llm_new_s1_summary.csv        ← skips API in Cell 24
llm_new_s2_raw.csv            ← skips API in Cell 24
llm_new_s3_engineered.csv     ← skips API in Cell 24
```

### With API key (full re-run)
```bash
pip install anthropic pandas numpy scikit-learn xgboost matplotlib seaborn shap Pillow

# Set your key in Cell 0:
ANTHROPIC_API_KEY = "sk-ant-..."
```

---

## Key Findings

1. **LLM outperforms ML by 35% AUC** — raw text context beats structured features for this task
2. **Feature engineering hurts LLM performance** — S3 (engineered) performs worse than S2 (raw), confirming LLMs read text better than tables
3. **Recommended hybrid:** ML ranks all 331 products for efficiency; LLM evaluates top candidates for final selection
4. **SHAP analysis** shows price x quality interaction is the strongest ML predictor, followed by sleeve type, material, and color

---

## Tech Stack

- **LLM:** Claude API (Anthropic) — multimodal image analysis + buyer agent classification
- **ML:** XGBoost, Random Forest (scikit-learn)
- **Explainability:** SHAP
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn
