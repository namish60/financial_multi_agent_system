import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from tools import fetch_stock_data, fetch_news_and_sentiment

# 1. Load API Keys
load_dotenv()

# 2. Initialize the "Brain" (LLM) with Google Gemini
llm = LLM(
    model="models/gemini-2.5-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    provider="gemini",
    max_tokens=512
)

# 3. Define the Financial Analyst Agent
fin_analyst = Agent(
    role='Senior Financial Analyst',
    goal='Analyze {ticker} stock performance and provide a summary of its financial health.',
    backstory="""You are a seasoned Wall Street analyst. You excel at looking at 
    fundamental ratios and market data to determine if a company is healthy or struggling.""",
    tools=[fetch_stock_data],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# 4. Define the News & Sentiment Agent
news_sentiment_analyst = Agent(
    role="Market News & Sentiment Analyst",
    goal="Analyze latest news for {ticker} and determine overall market sentiment.",
    backstory="""You are a financial news expert who tracks market-moving headlines
    and investor sentiment to assess how news may impact a stock.""",
    tools=[fetch_news_and_sentiment],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# 5. Define Tasks

analysis_task = Task(
    description="""
        Use the fetch_stock_data tool to get financial metrics for {ticker}.
        Analyze the P/E ratio and revenue growth.
        Compare them to general industry standards.
        Provide a final summary of the company's financial state.
    """,
    expected_output="A 3-paragraph financial summary of the stock including key metrics and a health assessment.",
    agent=fin_analyst
)

news_task = Task(
    description="""
        Use the fetch_news_and_sentiment tool to get recent news for {ticker}.
        Analyze the tone of headlines and descriptions.
        Classify overall sentiment as Positive, Neutral, or Negative.
        Summarize how news may impact the stock.
    """,
    expected_output="A sentiment report with key headlines and a sentiment classification.",
    agent=news_sentiment_analyst
)

# 6. The Orchestrator (The Crew)
financial_crew = Crew(
    agents=[fin_analyst, news_sentiment_analyst],
    tasks=[analysis_task, news_task],
    process=Process.sequential
)

# 7. Execute!
print("### Starting the Financial + News Sentiment Analysis ###")

input_ticker = input("enter a comapany ticker ")
result = financial_crew.kickoff(inputs={'ticker': input_ticker})

print("\n\n########################")
print("## FINAL REPORT ##")
print("########################\n")
print(result)