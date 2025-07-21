# 🛒 SmartMarket CLI - ML Powered Marketplace

A complete e-commerce CLI application demonstrating 4 core machine learning concepts:

- **Regression**: Price Prediction & Sales Forecasting
- **Classification**: Customer Satisfaction Prediction
- **Clustering**: Customer Segmentation
- **Association Rules**: Product Recommendations

## Features
- Generate or load realistic marketplace data (products, customers, transactions)
- Predict delivery time and sales volume
- Predict customer satisfaction
- Segment customers into behavioral groups
- Discover product associations and make recommendations
- Ask natural language questions about your data (powered by PandasAI + OpenAI)
- Predict sales for any category and month
- Save/load trained models
- Docker support for easy deployment

## Setup

### 1. Clone the Repository
```bash
# Clone this repo and cd into it
cd AiMl
```

### 2. Install Python Dependencies
It is recommended to use a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
# or
source .venv/bin/activate  # On Mac/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Prepare Data
- By default, the app will generate synthetic data on first run.
- To use real Olist e-commerce data, place the Olist CSVs in a `data/` folder and run `prepare_olist_data.py` to generate `products.csv`, `customers.csv`, and `transactions.csv`.

### 4. Run the CLI
```bash
python marketplace_ml.py
```

## Docker Usage

### 1. Build the Docker Image
```bash
docker build -t smartmarket-cli .
```

### 2. Run the CLI in Docker
```bash
docker run -it --rm smartmarket-cli
```

- For OpenAI-powered features, pass your API key:
  ```bash
  docker run -it --rm -e OPENAI_API_KEY=sk-... smartmarket-cli
  ```
- To persist data, mount the data folder:
  ```bash
  docker run -it --rm -v ${PWD}/data:/app/data smartmarket-cli
  ```

## Natural Language Q&A (PandasAI)
- Ask questions like "What were the sales in March?", "Top customer by revenue", "Plot sales trend".
- Requires an OpenAI API key (set as `OPENAI_API_KEY` environment variable or enter at prompt).
- For large datasets, queries may be slow or memory-intensive.

## Customization
- Edit `requirements.txt` to add/remove dependencies.
- Modify `prepare_olist_data.py` to change data preparation logic.
- Extend `marketplace_ml.py` to add new ML features or menu options.

## Troubleshooting
- If you see errors about missing packages, run `pip install -r requirements.txt` again.
- For Docker issues, ensure you use `-it` for interactive mode.
- For PandasAI/OpenAI errors, check your API key and internet connection.

## License
MIT License

---

**SmartMarket CLI** is an educational project for learning applied machine learning in a realistic e-commerce context. Contributions and suggestions welcome! 