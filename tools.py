import os
import requests
import yfinance as yf
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

@tool("fetch_stock_data")
def fetch_stock_data(ticker: str = "NVDA"):
    """
    Fetches historical price, market cap, P/E ratio, and revenue growth for a given stock ticker.
    Default ticker is NVDA to avoid CrewAI tool validation errors.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        data = {
            "price": info.get("currentPrice"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "revenue_growth": info.get("revenueGrowth"),
            "ebitda_margins": info.get("ebitdaMargins"),
            "currency": info.get("currency"),
        }

        return data

    except Exception as e:
        return {
            "error": f"Error fetching data for {ticker}",
            "details": str(e)
        }


@tool("fetch_news_and_sentiment")
def fetch_news_and_sentiment(ticker: str = "NVDA"):
    """
    Fetches 2 to 3 latest news headlines for a stock ticker using NewsAPI
    and returns basic sentiment-ready text.
    """
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": ticker,
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": NEWS_API_KEY,
            "pageSize": 5
        }

        response = requests.get(url, params=params)
        data = response.json()

        if data.get("status") != "ok":
            return {"error": "Failed to fetch news", "details": data}

        articles = data.get("articles", [])
        headlines = [
            {
                "title": a["title"],
                "description": a["description"],
                "source": a["source"]["name"]
            }
            for a in articles
        ]

        return {
            "ticker": ticker,
            "news_count": len(headlines),
            "headlines": headlines
        }

    except Exception as e:
        return {
            "error": f"Error fetching news for {ticker}",
            "details": str(e)
        }