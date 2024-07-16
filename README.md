# zylu-task

# Customer Prediction API

This FastAPI application predicts the likelihood of a customer returning to a store and repurchasing products. It also recommends products to increase store revenue.

## Features

- Predict Return Likelihood
- Predict Repurchase Likelihood

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- Joblib
- Scikit-learn
- Pandas

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/customer-prediction-api.git
    cd customer-prediction-api
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Training the Models

1. Place your dataset in the `data` folder (e.g., `purchase_history.csv`).

2. Train and save the models


## Running the API

1. Run the FastAPI server:

    ```bash
    uvicorn main:app --reload
    ```

2. Test the endpoints:
    - `POST /predict_return`
    - `POST /predict_repurchase`
