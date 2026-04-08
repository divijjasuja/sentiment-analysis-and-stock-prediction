# Stock Price Prediction with Sentiment Analysis

Predicting NVIDIA's stock price by combining Twitter sentiment analysis with time-series forecasting — exploring whether social media signals can improve financial prediction models.

## Overview

The stock market is influenced by more than just numbers — public opinion, news, and social media play an increasingly significant role. This project investigates whether Twitter-derived sentiment can improve the accuracy of LSTM-based stock price predictions for NVIDIA Corporation (NASDAQ: NVDA).

Two LSTM models were built and compared:
- **Baseline** — trained on historical stock price data only
- **Sentiment-enhanced** — trained on stock data combined with RoBERTa sentiment scores from scraped tweets

---

## Dataset

- **Stock Data** — NVIDIA (NVDA) historical price data for Q1 2024, sourced from Yahoo Finance
- **Tweet Data** — NVIDIA-related tweets scraped using the `twscrape` library via asynchronous API calls

---

## Methodology

### 1. Tweet Preprocessing
Raw tweets were cleaned through:
- Lowercasing, link and emoji removal
- Tokenization and stopword filtering
- Lemmatization

### 2. Sentiment Analysis
Sentiment classification was performed using **RoBERTa** — a transformer model pre-trained on over 124 million tweets. Each tweet was classified as:
- `Positive` → mapped to **1**
- `Neutral` → mapped to **0**
- `Negative` → mapped to **-1**

Sentiment scores were aggregated by date and merged with daily stock data.

### 3. Feature Engineering
The unified dataset included:
- Opening price, closing price, trading volume
- Average daily sentiment score
- Daily tweet volume

### 4. LSTM Forecasting
Both models used:
- 80/20 train-test split
- MinMax scaling for normalization
- 5-day sliding time windows to predict next-day price
- 512 training epochs
- MSE as the loss function

---

## Results

### Model Performance
Both models converged to the same MSE on the test set, suggesting that sentiment data alone did not improve overall predictive accuracy. Training loss stabilized after approximately 100–200 epochs in both cases, with no significant gains beyond that point.

### Correlation Analysis

| Variable Pair | Pearson Correlation | Interpretation |
|---|---|---|
| Tweet Volume vs Stock Trading Volume | **0.72** | Strong positive correlation |
| Tweet Sentiment vs Closing Price | **0.30** | Weak positive correlation |

- The strong **0.72** correlation between tweet volume and trading volume suggests that social media activity closely tracks real market interest and may serve as a leading indicator of trading behavior
- A notable case: a sharp sentiment spike on **Feb 20–22, 2024** coincided with a substantial increase in NVIDIA's stock price, hinting at short-term sentiment-driven price movement

### Key Takeaway
While sentiment did not improve quantitative MSE performance, correlation analysis and case-specific trends reveal that sentiment may offer valuable supplementary signals — particularly during volatile or high-activity market periods.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch | LSTM model implementation and training |
| RoBERTa (HuggingFace) | Sentiment classification |
| twscrape | Twitter data scraping |
| Yahoo Finance API | Historical stock data |
| NumPy / Pandas | Data processing |
| Matplotlib | Visualization |

---

## Project Structure

```
├── Divij_Jasuja_Bachelors_Project_Stock_prediction.ipynb   # Main notebook
├── stock_data/                                             # Stock data
├── twitter_data/                                           # twitter data
├── requirements.txt                                        # Dependencies
├── NVDA_final                                              # csv file with open, close, vol, etc. + sentiment sum 
└── pyproject.toml                                          # uv toml file
```

---

## Installation

```bash
# Install uv (if not already installed)
pip install uv

# Create and activate virtual environment
uv venv .venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv sync
```

---

## Future Work

- Expand dataset to cover longer timeframes (1 year or more)
- Filter out low-volume sentiment days as noise
- Explore alternative architectures — GRU, Transformer-based models
- Combine sentiment with additional market indicators (e.g. trading volume, moving averages)
- Reduce training epochs to 100–200 based on observed convergence behavior

---

## References

- Ray et al. (2021) — Hybrid BST-LSTM for news sentiment stock forecasting, IEEE TCSS
- Alam et al. (2024) — LSTM-DNN hybrid model for stock prediction, IEEE Access
- Ren et al. (2018) — Sentiment-based SVM for stock market direction, IEEE Systems Journal
- Minh et al. (2018) — Two-stream GRU for short-term stock trend prediction, IEEE Access
- Bacco et al. (2024) — LSTM + Twitter sentiment for bank stock prediction, IEEE Access