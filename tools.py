# import os
# import requests
# import yfinance as yf
# import numpy as np
# from crewai.tools import tool
# from dotenv import load_dotenv

# load_dotenv()

# NEWS_API_KEY = os.getenv("NEWS_API_KEY")


# @tool("fetch_stock_data")
# def fetch_stock_data(ticker: str = "NVDA"):
#     """
#     Fetches financial metrics such as price, market cap, P/E ratio,
#     revenue growth, and EBITDA margins for a given stock ticker
#     using Yahoo Finance.
#     """
#     try:
#         stock = yf.Ticker(ticker)
#         info = stock.info

#         data = {
#             "price": info.get("currentPrice"),
#             "market_cap": info.get("marketCap"),
#             "pe_ratio": info.get("trailingPE"),
#             "revenue_growth": info.get("revenueGrowth"),
#             "ebitda_margins": info.get("ebitdaMargins"),
#             "currency": info.get("currency"),
#         }

#         return data

#     except Exception as e:
#         return {
#             "error": f"Error fetching data for {ticker}",
#             "details": str(e)
#         }


# @tool("fetch_news_and_sentiment")
# def fetch_news_and_sentiment(ticker: str = "NVDA"):
#     """
#     Fetches the latest news headlines for a stock ticker using NewsAPI
#     and returns sentiment-ready text data including titles,
#     descriptions, and sources.
#     """
#     try:
#         url = "https://newsapi.org/v2/everything"
#         params = {
#             "q": ticker,
#             "sortBy": "publishedAt",
#             "language": "en",
#             "apiKey": NEWS_API_KEY,
#             "pageSize": 5
#         }

#         response = requests.get(url, params=params)
#         data = response.json()

#         if data.get("status") != "ok":
#             return {"error": "Failed to fetch news", "details": data}

#         articles = data.get("articles", [])
#         headlines = [
#             {
#                 "title": a.get("title"),
#                 "description": a.get("description"),
#                 "source": a.get("source", {}).get("name")
#             }
#             for a in articles
#         ]

#         return {
#             "ticker": ticker,
#             "news_count": len(headlines),
#             "headlines": headlines
#         }

#     except Exception as e:
#         return {
#             "error": f"Error fetching news for {ticker}",
#             "details": str(e)
#         }


# @tool("fetch_risk_metrics")
# def fetch_risk_metrics(ticker: str = "NVDA"):
#     """
#     Fetches quantitative risk metrics for a given stock ticker
#     including annualized volatility, beta, and maximum drawdown
#     using historical price data from Yahoo Finance.
#     """
#     try:
#         stock = yf.Ticker(ticker)
#         hist = stock.history(period="1y")

#         returns = hist["Close"].pct_change().dropna()
#         volatility = np.std(returns) * np.sqrt(252)

#         info = stock.info
#         beta = info.get("beta")

#         rolling_max = hist["Close"].cummax()
#         drawdown = (hist["Close"] - rolling_max) / rolling_max
#         max_drawdown = drawdown.min()

#         return {
#             "volatility": round(float(volatility), 4),
#             "beta": beta,
#             "max_drawdown": round(float(max_drawdown), 4)
#         }

#     except Exception as e:
#         return {
#             "error": f"Error fetching risk data for {ticker}",
#             "details": str(e)
#         }


#NEW CODE ============================================================
import os
import requests
import yfinance as yf
import numpy as np
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")


@tool("fetch_stock_data")
def fetch_stock_data(ticker: str):
    """Fetch stock price, valuation, and growth metrics."""
    stock = yf.Ticker(ticker)
    info = stock.info

    return {
        "price": info.get("currentPrice"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "revenue_growth": info.get("revenueGrowth"),
        "ebitda_margins": info.get("ebitdaMargins"),
        "currency": info.get("currency"),
    }


@tool("fetch_news_and_sentiment")
def fetch_news_and_sentiment(ticker: str):
    """Fetch latest news headlines for sentiment analysis."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWS_API_KEY,
        "pageSize": 5
    }

    response = requests.get(url, params=params).json()
    articles = response.get("articles", [])

    return [
        {
            "title": a.get("title"),
            "description": a.get("description"),
            "source": a.get("source", {}).get("name"),
        }
        for a in articles
    ]


@tool("fetch_risk_metrics")
def fetch_risk_metrics(ticker: str):
    """Compute volatility, beta, and max drawdown."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")

    returns = hist["Close"].pct_change().dropna()
    volatility = np.std(returns) * np.sqrt(252)

    rolling_max = hist["Close"].cummax()
    drawdown = (hist["Close"] - rolling_max) / rolling_max

    return {
        "volatility": round(float(volatility), 4),
        "beta": stock.info.get("beta"),
        "max_drawdown": round(float(drawdown.min()), 4),
    }
