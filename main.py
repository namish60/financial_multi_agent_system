import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from tools import fetch_stock_data  

# 1. Load API Keys
load_dotenv()

# 2. Initialize the "Brain" (LLM) with Google Gemini
llm = LLM(
    model="models/gemini-2.5-flash",  # Updated model
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    provider="gemini",        # correct provider
    max_tokens=512
)

# 3. Define the Financial Analyst Agent
fin_analyst = Agent(
    role='Senior Financial Analyst',
    goal='Analyze {ticker} stock performance and provide a summary of its financial health.',
    backstory="""You are a seasoned Wall Street analyst. You excel at looking at 
    fundamental ratios and market data to determine if a company is healthy or struggling.""",
    tools=[fetch_stock_data],  # Use the function name directly
    llm=llm,
    verbose=True, 
    allow_delegation=False
)

# 4. Define the Task
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

# 5. The Orchestrator (The Crew)
financial_crew = Crew(
    agents=[fin_analyst],
    tasks=[analysis_task],
    process=Process.sequential 
)

# 6. Execute!
print("### Starting the Financial Analysis ###")
result = financial_crew.kickoff(inputs={'ticker': 'NVDA'}) 

print("\n\n########################")
print("## FINAL REPORT ##")
print("########################\n")
print(result)