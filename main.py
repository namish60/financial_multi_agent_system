import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from tools import fetch_stock_data, fetch_news_and_sentiment, fetch_risk_metrics
from rapidfuzz import fuzz

load_dotenv()

# Initialize LLM
llm = LLM(
    model="models/gemini-2.5-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    provider="gemini",
    max_tokens=512
)

# =========================
# Agents
# =========================

fin_analyst = Agent(
    role='Senior Financial Analyst',
    goal='Analyze {ticker} stock performance and provide a summary of its financial health.',
    backstory="""You are a seasoned Wall Street analyst focused on
    company fundamentals and valuation metrics.""",
    tools=[fetch_stock_data],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

news_sentiment_analyst = Agent(
    role="Market News & Sentiment Analyst",
    goal="Analyze latest news for {ticker} and determine overall market sentiment.",
    backstory="""You track market-moving headlines and investor sentiment
    to assess how news impacts stock prices.""",
    tools=[fetch_news_and_sentiment],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

risk_analyst = Agent(
    role="Risk Assessment Analyst",
    goal="Assess market risk for {ticker} using volatility, beta, and drawdowns.",
    backstory="""You are a quantitative risk analyst who evaluates
    downside risk and stock stability.""",
    tools=[fetch_risk_metrics],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# =========================
# Tasks
# =========================

analysis_task = Task(
    description="""
        Use fetch_stock_data to get financial metrics for {ticker}.
        Analyze the P/E ratio and revenue growth.
        Provide a financial health summary.
    """,
    expected_output="A financial summary with key metrics and health assessment.",
    agent=fin_analyst
)

news_task = Task(
    description="""
        Use fetch_news_and_sentiment to get recent news for {ticker}.
        Analyze sentiment and summarize impact on the stock.
    """,
    expected_output="A sentiment report with key headlines and classification.",
    agent=news_sentiment_analyst
)

risk_task = Task(
    description="""
        Use fetch_risk_metrics to get volatility, beta, and max drawdown for {ticker}.
        Interpret these metrics and classify overall risk as Low, Medium, or High.
    """,
    expected_output="A risk report with volatility, beta, drawdown, and risk classification.",
    agent=risk_analyst
)

# =========================
# Fuzzy Matching Utility
# =========================

def fuzzy_contains(query: str, keywords: list, threshold: int = 75) -> bool:
    q = query.lower()
    for kw in keywords:
        score = fuzz.partial_ratio(q, kw.lower())
        if score >= threshold:
            return True
    return False

# =========================
# Rule-Based + Fuzzy Orchestrator
# =========================

def orchestrate(user_query: str):
    q = user_query.lower()

    selected_agents = []
    selected_tasks = []

    # Intent keyword banks
    financial_keywords = [
        "financial", "fundamental", "valuation", "pe ratio",
        "revenue", "growth", "earnings", "profit", "balance sheet"
    ]

    news_keywords = [
        "news", "sentiment", "headline", "media",
        "press", "story", "market buzz", "investor mood"
    ]

    risk_keywords = [
        "risk", "volatile", "volatility", "drawdown",
        "beta", "stability", "danger", "crash", "loss"
    ]

    # --- Financial intent ---
    if fuzzy_contains(q, financial_keywords, threshold=70):
        selected_agents.append(fin_analyst)
        selected_tasks.append(analysis_task)

    # --- News / Sentiment intent ---
    if fuzzy_contains(q, news_keywords, threshold=70):
        selected_agents.append(news_sentiment_analyst)
        selected_tasks.append(news_task)

    # --- Risk intent ---
    if fuzzy_contains(q, risk_keywords, threshold=70):
        selected_agents.append(risk_analyst)
        selected_tasks.append(risk_task)

    # --- Fallback: full analysis ---
    if not selected_agents:
        print("\n[Orchestrator] No clear intent detected → Running full analysis\n")
        selected_agents = [fin_analyst, news_sentiment_analyst, risk_analyst]
        selected_tasks = [analysis_task, news_task, risk_task]

    return selected_agents, selected_tasks

# =========================
# Run System
# =========================

print("### Multi-Agent Financial Analysis System ###")

user_query = input("\nAsk your financial question: ")
input_ticker = input("Enter a company ticker: ")

agents, tasks = orchestrate(user_query)

print("\n[Orchestrator] Selected Agents:")
for a in agents:
    print(f" - {a.role}")

financial_crew = Crew(
    agents=agents,
    tasks=tasks,
    process=Process.sequential
)

result = financial_crew.kickoff(inputs={'ticker': input_ticker})

print("\n\n########################")
print("## FINAL REPORT ##")
print("########################\n")
print(result)