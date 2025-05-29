# Stock Market Predictor Chatbot

An AI-powered chatbot that provides stock market predictions, price information, market sentiment analysis, and news recommendations using machine learning.

## Features

- **Stock Price Information**: Get the current price of various stocks
- **Stock Movement Predictions**: AI-powered predictions on whether a stock will go up or down
- **Market Sentiment Analysis**: Understand the current market sentiment
- **Stock News Recommendations**: Get recommendations for where to find the latest news on specific stocks

## Supported Stocks

- Apple (AAPL)
- Microsoft (MSFT)
- Google/Alphabet (GOOGL)
- Amazon (AMZN)
- Meta/Facebook (META)
- Tesla (TSLA)
- NVIDIA (NVDA)
- AMD (AMD)

## Technology Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Node.js, Express
- **Machine Learning**: Python, scikit-learn, pandas, numpy, yfinance

## Installation

### Prerequisites

- Node.js (v14 or higher)
- Python 3.7+ with pip

### Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd stock-chatbot
   ```

2. Install Node.js dependencies:
   ```
   npm install
   ```

3. Install Python dependencies:
   ```
   npm run install-py-deps
   ```

4. Train the machine learning models:
   ```
   npm run train
   ```

## Usage

1. Start the server:
   ```
   npm start
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Start chatting with the bot by asking questions about stocks!

## Example Queries

- "What is the current price of AAPL?"
- "Will MSFT go up tomorrow?"
- "How is the market sentiment today?"
- "Any news about TSLA?"
- "Should I buy NVDA stock?"
- "What's the prediction for GOOGL?"

## Development

For development with auto-restart:
```
npm run dev
```

## License

ISC