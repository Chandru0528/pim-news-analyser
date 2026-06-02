import os
import json
from anthropic import Anthropic
from pinecone import Pinecone
from newsapi import NewsApiClient
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

INDEX_NAME = "financial-news"

def search_memory(query: str, top_k: int = 3) -> dict:
    """Search Pinecone for similar past articles"""
    try:
        index = pc.Index(INDEX_NAME)
        results = index.query(
            vector=[0.1] * 1536,
            top_k=top_k,
            include_metadata=True
        )
        matches = []
        for match in results.matches:
            matches.append({
                "headline": match.metadata.get("headline", ""),
                "sentiment": match.metadata.get("sentiment", ""),
                "risk_level": match.metadata.get("risk_level", 0),
                "stock_impact": match.metadata.get("stock_impact", ""),
                "sectors": match.metadata.get("sectors", ""),
                "score": round(match.score, 3)
            })
        return {
            "status": "success",
            "query": query,
            "matches_found": len(matches),
            "past_articles": matches
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def fetch_related_news(topic: str, max_articles: int = 3) -> dict:
    """Fetch related headlines from NewsAPI"""
    try:
        response = newsapi.get_everything(
            q=topic,
            language="en",
            sort_by="relevancy",
            page_size=max_articles
        )
        articles = []
        for article in response.get("articles", [])[:max_articles]:
            articles.append({
                "headline": article.get("title", ""),
                "source": article.get("source", {}).get("name", ""),
                "published_at": article.get("publishedAt", "")
            })
        return {
            "status": "success",
            "topic": topic,
            "articles_found": len(articles),
            "related_articles": articles
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def send_alert(message: str, risk_level: int, headline: str) -> dict:
    """Send alert for high risk news"""
    try:
        alert = {
            "status": "alert_sent",
            "risk_level": risk_level,
            "headline": headline,
            "message": message,
            "channel": "#financial-alerts",
            "priority": "HIGH" if risk_level >= 4 else "MEDIUM"
        }
        print(f"\n🚨 ALERT TRIGGERED: {message}\n")
        return alert
    except Exception as e:
        return {"status": "error", "message": str(e)}

TOOLS_DEFINITION = [
    {
        "name": "search_memory",
        "description": "Search Pinecone vector database for similar past financial news articles. Use this to find historical context and patterns before finalizing your analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query related to the financial topic"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of past articles to retrieve (default 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_related_news",
        "description": "Fetch related financial news headlines from NewsAPI. Use this when you need more context about a topic before making a risk assessment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic or keywords to search for related news"
                },
                "max_articles": {
                    "type": "integer",
                    "description": "Maximum number of articles to fetch (default 3)",
                    "default": 3
                }
            },
            "required": ["topic"]
        }
    },
    {
        "name": "send_alert",
        "description": "Send an alert for high risk financial news. Only use this when risk_level is 4 or 5.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Alert message summarizing the risk"
                },
                "risk_level": {
                    "type": "integer",
                    "description": "Risk level from 1 to 5"
                },
                "headline": {
                    "type": "string",
                    "description": "The original headline that triggered the alert"
                }
            },
            "required": ["message", "risk_level", "headline"]
        }
    }
]

TOOL_MAP = {
    "search_memory": search_memory,
    "fetch_related_news": fetch_related_news,
    "send_alert": send_alert
}
