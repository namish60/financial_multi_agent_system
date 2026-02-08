# import os
# from dotenv import load_dotenv
# from crewai import Agent, Task, Crew, Process, LLM
# from tools import fetch_stock_data, fetch_news_and_sentiment, fetch_risk_metrics
# from rapidfuzz import fuzz

# load_dotenv()

# # Initialize LLM
# llm = LLM(
#     model="models/gemini-2.5-flash-lite",
#     api_key=os.getenv("GOOGLE_API_KEY"),
#     temperature=0,
#     provider="gemini",
#     max_tokens=512
# )

# # =========================
# # Agents
# # =========================

# fin_analyst = Agent(
#     role='Senior Financial Analyst',
#     goal='Analyze {ticker} stock performance and provide a summary of its financial health.',
#     backstory="""You are a seasoned Wall Street analyst focused on
#     company fundamentals and valuation metrics.""",
#     tools=[fetch_stock_data],
#     llm=llm,
#     verbose=True,
#     allow_delegation=False
# )

# news_sentiment_analyst = Agent(
#     role="Market News & Sentiment Analyst",
#     goal="Analyze latest news for {ticker} and determine overall market sentiment.",
#     backstory="""You track market-moving headlines and investor sentiment
#     to assess how news impacts stock prices.""",
#     tools=[fetch_news_and_sentiment],
#     llm=llm,
#     verbose=True,
#     allow_delegation=False
# )

# risk_analyst = Agent(
#     role="Risk Assessment Analyst",
#     goal="Assess market risk for {ticker} using volatility, beta, and drawdowns.",
#     backstory="""You are a quantitative risk analyst who evaluates
#     downside risk and stock stability.""",
#     tools=[fetch_risk_metrics],
#     llm=llm,
#     verbose=True,
#     allow_delegation=False
# )

# # =========================
# # Tasks
# # =========================

# analysis_task = Task(
#     description="""
#         Use fetch_stock_data to get financial metrics for {ticker}.
#         Analyze the P/E ratio and revenue growth.
#         Provide a financial health summary.
#     """,
#     expected_output="A financial summary with key metrics and health assessment.",
#     agent=fin_analyst
# )

# news_task = Task(
#     description="""
#         Use fetch_news_and_sentiment to get recent news for {ticker}.
#         Analyze sentiment and summarize impact on the stock.
#     """,
#     expected_output="A sentiment report with key headlines and classification.",
#     agent=news_sentiment_analyst
# )

# risk_task = Task(
#     description="""
#         Use fetch_risk_metrics to get volatility, beta, and max drawdown for {ticker}.
#         Interpret these metrics and classify overall risk as Low, Medium, or High.
#     """,
#     expected_output="A risk report with volatility, beta, drawdown, and risk classification.",
#     agent=risk_analyst
# )

# # =========================
# # Fuzzy Matching Utility
# # =========================

# def fuzzy_contains(query: str, keywords: list, threshold: int = 75) -> bool:
#     q = query.lower()
#     for kw in keywords:
#         score = fuzz.partial_ratio(q, kw.lower())
#         if score >= threshold:
#             return True
#     return False

# # =========================
# # Rule-Based + Fuzzy Orchestrator
# # =========================

# def orchestrate(user_query: str):
#     q = user_query.lower()

#     selected_agents = []
#     selected_tasks = []

#     # Intent keyword banks
#     financial_keywords = [
#         "financial", "fundamental", "valuation", "pe ratio",
#         "revenue", "growth", "earnings", "profit", "balance sheet"
#     ]

#     news_keywords = [
#         "news", "sentiment", "headline", "media",
#         "press", "story", "market buzz", "investor mood"
#     ]

#     risk_keywords = [
#         "risk", "volatile", "volatility", "drawdown",
#         "beta", "stability", "danger", "crash", "loss"
#     ]

#     # --- Financial intent ---
#     if fuzzy_contains(q, financial_keywords, threshold=70):
#         selected_agents.append(fin_analyst)
#         selected_tasks.append(analysis_task)

#     # --- News / Sentiment intent ---
#     if fuzzy_contains(q, news_keywords, threshold=70):
#         selected_agents.append(news_sentiment_analyst)
#         selected_tasks.append(news_task)

#     # --- Risk intent ---
#     if fuzzy_contains(q, risk_keywords, threshold=70):
#         selected_agents.append(risk_analyst)
#         selected_tasks.append(risk_task)

#     # --- Fallback: full analysis ---
#     if not selected_agents:
#         print("\n[Orchestrator] No clear intent detected → Running full analysis\n")
#         selected_agents = [fin_analyst, news_sentiment_analyst, risk_analyst]
#         selected_tasks = [analysis_task, news_task, risk_task]

#     return selected_agents, selected_tasks

# # =========================
# # Run System
# # =========================

# print("### Multi-Agent Financial Analysis System ###")

# user_query = input("\nAsk your financial question: ")
# input_ticker = input("Enter a company ticker: ")

# agents, tasks = orchestrate(user_query)

# print("\n[Orchestrator] Selected Agents:")
# for a in agents:
#     print(f" - {a.role}")

# financial_crew = Crew(
#     agents=agents,
#     tasks=tasks,
#     process=Process.sequential
# )

# result = financial_crew.kickoff(inputs={'ticker': input_ticker})

# print("\n\n########################")
# print("## FINAL REPORT ##")
# print("########################\n")
# print(result)


# # NEW CODE WITH LANGRAPH====================================================
# import os
# from typing import TypedDict, List, Annotated
# from dotenv import load_dotenv
# from crewai import Agent, Task, Crew, Process, LLM
# from langgraph.graph import StateGraph, END, add_messages
# from rapidfuzz import fuzz

# from tools import (
#     fetch_stock_data,
#     fetch_news_and_sentiment,
#     fetch_risk_metrics
# )

# # =========================
# # Environment
# # =========================
# load_dotenv()

# # =========================
# # LLM
# # =========================
# llm = LLM(
#     model="models/gemini-2.5-flash",
#     api_key=os.getenv("GOOGLE_API_KEY"),
#     provider="gemini",
#     temperature=0,
#     max_tokens=512
# )

# # =========================
# # Agents
# # =========================
# fin_analyst = Agent(
#     role="Senior Financial Analyst",
#     goal="Analyze {ticker} financial health",
#     backstory="Expert in valuation and fundamentals",
#     tools=[fetch_stock_data],
#     llm=llm,
#     allow_delegation=False,
#     verbose=False
# )

# news_analyst = Agent(
#     role="News & Sentiment Analyst",
#     goal="Analyze 3 latest news and sentiment for {ticker}",
#     backstory="Tracks market-moving headlines",
#     tools=[fetch_news_and_sentiment],
#     llm=llm,
#     allow_delegation=False,
#     verbose=False
# )

# risk_analyst = Agent(
#     role="Risk Analyst",
#     goal="Assess market risk for {ticker}",
#     backstory="Quantitative risk specialist",
#     tools=[fetch_risk_metrics],
#     llm=llm,
#     allow_delegation=False,
#     verbose=False
# )

# # =========================
# # Tasks
# # =========================
# financial_task = Task(
#     description="Analyze valuation, P/E ratio, and growth for {ticker}",
#     expected_output="Financial health summary",
#     agent=fin_analyst
# )

# news_task = Task(
#     description="Analyze recent news and sentiment for {ticker}",
#     expected_output="Sentiment analysis report",
#     agent=news_analyst
# )

# risk_task = Task(
#     description="Analyze volatility, beta, and drawdown for {ticker}",
#     expected_output="Risk classification report",
#     agent=risk_analyst
# )

# # =========================
# # LangGraph State
# # =========================
# class GraphState(TypedDict):
#     query: str
#     ticker: str
#     routes: List[str]
#     outputs: Annotated[List[str], add_messages]

# # =========================
# # Intent Detection
# # =========================
# def fuzzy_contains(query: str, keywords: list, threshold: int = 70) -> bool:
#     for kw in keywords:
#         if fuzz.partial_ratio(query, kw.lower()) >= threshold:
#             return True
#     return False

# financial_keywords = [
#     "financial", "valuation", "pe",
#     "revenue", "growth", "earnings"
# ]

# news_keywords = [
#     "news", "sentiment", "headline",
#     "media", "market"
# ]

# risk_keywords = [
#     "risk", "volatility", "beta",
#     "drawdown", "loss"
# ]

# # =========================
# # Orchestrator
# # =========================
# def orchestrator_node(state: GraphState):
#     q = state["query"].lower()
#     routes = []

#     if fuzzy_contains(q, financial_keywords):
#         routes.append("financial")

#     if fuzzy_contains(q, news_keywords):
#         routes.append("news")

#     if fuzzy_contains(q, risk_keywords):
#         routes.append("risk")

#     if not routes:
#         routes = ["financial", "news", "risk"]

#     return {"routes": routes}

# # =========================
# # Agent Nodes
# # =========================
# def financial_node(state: GraphState):
#     crew = Crew(
#         agents=[fin_analyst],
#         tasks=[financial_task],
#         process=Process.sequential,
#         verbose=False
#     )
#     result = crew.kickoff(inputs={"ticker": state["ticker"]})
#     return {"outputs": [f"📊 FINANCIAL ANALYSIS\n{result}"]}

# def news_node(state: GraphState):
#     crew = Crew(
#         agents=[news_analyst],
#         tasks=[news_task],
#         process=Process.sequential,
#         verbose=False
#     )
#     result = crew.kickoff(inputs={"ticker": state["ticker"]})
#     return {"outputs": [f"📰 NEWS & SENTIMENT\n{result}"]}

# def risk_node(state: GraphState):
#     crew = Crew(
#         agents=[risk_analyst],
#         tasks=[risk_task],
#         process=Process.sequential,
#         verbose=False
#     )
#     result = crew.kickoff(inputs={"ticker": state["ticker"]})
#     return {"outputs": [f"⚠️ RISK ASSESSMENT\n{result}"]}

# # =========================
# # Build Graph
# # =========================
# workflow = StateGraph(GraphState)

# workflow.add_node("orchestrator", orchestrator_node)
# workflow.add_node("financial", financial_node)
# workflow.add_node("news", news_node)
# workflow.add_node("risk", risk_node)

# workflow.set_entry_point("orchestrator")

# workflow.add_conditional_edges(
#     "orchestrator",
#     lambda state: state["routes"]
# )

# workflow.add_edge("financial", END)
# workflow.add_edge("news", END)
# workflow.add_edge("risk", END)

# app = workflow.compile()

# # =========================
# # Run
# # =========================
# print("\n### LangGraph + CrewAI Financial Multi-Agent System ###")

# query = input("\nAsk your financial question: ")
# ticker = input("Enter company ticker: ")

# state: GraphState = {
#     "query": query,
#     "ticker": ticker,
#     "routes": [],
#     "outputs": []
# }

# result = app.invoke(state)

# print("\n========================")
# print("FINAL REPORT")
# print("========================\n")

# for msg in result["outputs"]:
#     print(msg.content)
#     print()


import os
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM
from langgraph.graph import StateGraph, END, add_messages
from rapidfuzz import fuzz

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


from tools import (
    fetch_stock_data,
    fetch_news_and_sentiment,
    fetch_risk_metrics
)

# =========================
# Environment
# =========================
load_dotenv()

# =========================
# LLM
# =========================
llm = LLM(
    model="models/gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    provider="gemini",
    temperature=0,
    max_tokens=512
)

# =========================
# Agents
# =========================
fin_analyst = Agent(
    role="Senior Financial Analyst",
    goal="Analyze {ticker} financial health",
    backstory="Expert in valuation and fundamentals",
    tools=[fetch_stock_data],
    llm=llm,
    allow_delegation=False
)

news_analyst = Agent(
    role="News & Sentiment Analyst",
    goal="Analyze latest news for {ticker}",
    backstory="Tracks market-moving headlines",
    tools=[fetch_news_and_sentiment],
    llm=llm,
    allow_delegation=False
)

risk_analyst = Agent(
    role="Risk Analyst",
    goal="Assess market risk for {ticker}",
    backstory="Quantitative risk specialist",
    tools=[fetch_risk_metrics],
    llm=llm,
    allow_delegation=False
)

rag_agent = Agent(
    role="Document Intelligence Agent",
    goal="Answer questions strictly from uploaded documents",
    backstory="Expert at reading and extracting insights from documents",
    llm=llm,
    allow_delegation=False
)

# =========================
# Tasks
# =========================
financial_task = Task(
    description="Analyze valuation and growth for {ticker}",
    expected_output="Financial analysis",
    agent=fin_analyst
)

news_task = Task(
    description="Analyze news sentiment for {ticker}",
    expected_output="News sentiment report",
    agent=news_analyst
)

risk_task = Task(
    description="Analyze risk metrics for {ticker}",
    expected_output="Risk analysis",
    agent=risk_analyst
)

# =========================
# LangGraph State
# =========================
class GraphState(TypedDict):
    query: str
    ticker: str
    doc_path: str
    routes: List[str]
    outputs: Annotated[List[str], add_messages]

# =========================
# Intent Detection
# =========================
def fuzzy_contains(query: str, keywords: list) -> bool:
    return any(fuzz.partial_ratio(query, kw) > 70 for kw in keywords)

financial_keywords = ["financial", "valuation", "revenue", "growth"]
news_keywords = ["news", "sentiment", "headline"]
risk_keywords = ["risk", "volatility", "beta"]
doc_keywords = ["document", "pdf", "file", "report", "from document"]

# =========================
# Orchestrator
# =========================
def orchestrator_node(state: GraphState):
    q = state["query"].lower()
    routes = []

    if state["doc_path"]:
        routes.append("rag")

    if fuzzy_contains(q, financial_keywords):
        routes.append("financial")
    if fuzzy_contains(q, news_keywords):
        routes.append("news")
    if fuzzy_contains(q, risk_keywords):
        routes.append("risk")

    if not routes:
        routes = ["financial", "news", "risk"]

    return {"routes": routes}

# =========================
# Agent Nodes
# =========================
def financial_node(state: GraphState):
    crew = Crew(
        agents=[fin_analyst],
        tasks=[financial_task],
        process=Process.sequential
    )
    result = crew.kickoff(inputs={"ticker": state["ticker"]})
    return {"outputs": [f"📊 FINANCIAL ANALYSIS\n{result}"]}

def news_node(state: GraphState):
    crew = Crew(
        agents=[news_analyst],
        tasks=[news_task],
        process=Process.sequential
    )
    result = crew.kickoff(inputs={"ticker": state["ticker"]})
    return {"outputs": [f"📰 NEWS & SENTIMENT\n{result}"]}

def risk_node(state: GraphState):
    crew = Crew(
        agents=[risk_analyst],
        tasks=[risk_task],
        process=Process.sequential
    )
    result = crew.kickoff(inputs={"ticker": state["ticker"]})
    return {"outputs": [f"⚠️ RISK ASSESSMENT\n{result}"]}

# =========================
# RAG Node
# =========================
def rag_node(state: GraphState):
    path = state["doc_path"]

    # Load document
    loader = PyPDFLoader(path) if path.endswith(".pdf") else TextLoader(path)
    docs = loader.load()

    # Split document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    # Embeddings (sentence-transformer is PERFECT for RAG)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # New LangChain API
    retrieved_docs = retriever.invoke(state["query"])

    context = "\n\n".join(d.page_content for d in retrieved_docs)

    prompt = f"""
You are a document intelligence assistant.
Answer the question using ONLY the context below.
If the answer is not present, say: "Not found in the document."

Context:
{context}

Question:
{state['query']}
"""

    # ✅ CrewAI LLM call (NOT invoke)
    answer = rag_agent.llm.call(prompt)

    return {
        "outputs": [f"📄 DOCUMENT INSIGHTS\n{answer.strip()}"]
    }



# =========================
# Build Graph
# =========================
workflow = StateGraph(GraphState)

workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("financial", financial_node)
workflow.add_node("news", news_node)
workflow.add_node("risk", risk_node)
workflow.add_node("rag", rag_node)

workflow.set_entry_point("orchestrator")

workflow.add_conditional_edges("orchestrator", lambda s: s["routes"])

workflow.add_edge("financial", END)
workflow.add_edge("news", END)
workflow.add_edge("risk", END)
workflow.add_edge("rag", END)

app = workflow.compile()

# =========================
# Run
# =========================
print("\n### LangGraph + CrewAI Financial Multi-Agent System ###")

query = input("\nAsk your financial question: ")
ticker = input("Enter company ticker (or press Enter): ")
doc_path = input("Enter document path if any (or press Enter): ")

state: GraphState = {
    "query": query,
    "ticker": ticker,
    "doc_path": doc_path.strip(),
    "routes": [],
    "outputs": []
}

result = app.invoke(state)

print("\n========================")
print("FINAL REPORT")
print("========================\n")

for msg in result["outputs"]:
    if hasattr(msg, "content"):
        print(msg.content.strip())
        print()


