import yfinance as yf
from crewai.tools import tool

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