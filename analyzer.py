import os
import json
from anthropic import Anthropic
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "financial-news"

def init_pinecone():
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(INDEX_NAME)

def analyze_headline(headline: str) -> dict:
    prompt = f"""You are a senior financial analyst at a top investment bank.

Analyze this financial news headline and return a JSON object with exactly these fields:

{{
  "sentiment": "positive" | "negative" | "neutral",
  "sentiment_score": <float between -1.0 and 1.0>,
  "impacted_sectors": [<list of sectors as strings>],
  "stock_impact": "bullish" | "bearish" | "neutral",
  "risk_level": <integer 1 to 5>,
  "executive_summary": "<2-3 sentence summary>",
  "recommendations": "<2-3 actionable recommendations for a portfolio manager>"
}}

Headline: {headline}

Return only the JSON object. No explanation, no markdown, no extra text."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.content[0].text.strip()
    result = json.loads(raw)
    result["headline"] = headline
    return result

def init_pinecone():
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(INDEX_NAME)

def store_in_pinecone(analysis: dict):
    index = init_pinecone()
    import hashlib
    doc_id = hashlib.md5(analysis['headline'].encode()).hexdigest()
    index.upsert(vectors=[{
        "id": doc_id,
        "values": [0.1] * 1536,
        "metadata": {
            "headline": analysis["headline"],
            "sentiment": analysis["sentiment"],
            "risk_level": analysis["risk_level"],
            "stock_impact": analysis["stock_impact"],
            "sectors": ", ".join(analysis["impacted_sectors"])
        }
    }])
    return doc_id

def run_full_analysis(headline: str) -> dict:
    analysis = analyze_headline(headline)
    doc_id = store_in_pinecone(analysis)
    analysis["pinecone_id"] = doc_id
    return analysis
