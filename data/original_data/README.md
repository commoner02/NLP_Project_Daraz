# Bangla and Banglish E-commerce Reviews ABSA Dataset

## Overview

This dataset contains manually annotated Bangla and Banglish e-commerce product reviews collected from Daraz (Electronics category). It is developed for Aspect-Based Sentiment Analysis (ABSA) tasks in low-resource languages.

### Dataset Composition

| File                                           | Description                                               |
| ---------------------------------------------- | --------------------------------------------------------- |
| `1_raw_data/original_dataset.csv`              | Raw user reviews with metadata (rating, date, image URLs) |
| `2_preprocessed_data/preprocessed_dataset.csv` | Cleaned, normalized, and language-tagged reviews          |
| `3_annotated_data/annotated_dataset.csv`       | Aspect-level sentiment annotations for each review        |
| `metadata/statistics.csv`                      | Basic dataset-level statistics                            |
| `metadata/label_distribution.csv`              | Label frequency distribution                              |

### Statistics

- Total original reviews: 19638
- Total preprocessed reviews: 10657
- Total annotated reviews: 3587
- Language distribution: {'bn': 6731, 'banglish': 2797, 'mix': 1129}
- Total unique labels: 15

### Label Classes

product_quality_positive, price_positive, product_quality_negative, delivery_positive, product_quality_neutral, seller_service_positive, price_neutral, packaging_positive, delivery_negative, packaging_negative, price_negative, seller_service_negative, delivery_neutral, packaging_neutral, seller_service_neutral

### Annotation Schema

Each review may contain multiple (aspect, sentiment) pairs.

Example:

```
Text: ডেলিভারি দ্রুত ছিল কিন্তু প্রোডাক্ট কোয়ালিটি খারাপ।
Labels: delivery_positive#product_quality_negative
```

### Preprocessing Steps

1. Removed URLs, emails, and mentions
2. Converted emojis to text
3. Removed HTML entities
4. Normalized spacing and punctuation
5. Reduced repeated characters
6. Filtered non-Bangla reviews
7. Lowercased and standardized Banglish words
8. Removed reviews shorter than 5 words
9. Added language tags (`bn`, `banglish`, `mix`)

### Intended Use

- Aspect-Based Sentiment Classification
- Multilingual Sentiment Modeling
- Bangla-Banglish NLP Research

### License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

### Citation

If you use this dataset, please cite:

Shakil, Md Nurnabi; Talukder, Md. Ashraful Islam (2025), “Bangla and Banglish E-Commerce Reviews Dataset for Aspect-Based Sentiment Analysis”, Mendeley Data, V1, doi: 10.17632/n4n5y34p3s.1
