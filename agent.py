import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv
from tools import TOOLS_DEFINITION, TOOL_MAP

load_dotenv()

client = Anthropic()

SYSTEM_PROMPT = """You are an autonomous financial risk monitoring agent at a top investment bank.

Your goal: Given a financial news headline, produce a comprehensive risk assessment by reasoning across multiple steps.

You have access to 3 tools:
- search_memory: Search past articles in your memory for historical patterns
- fetch_related_news: Get related headlines for more context
- send_alert: Trigger an alert (ONLY when risk_level is 4 or 5)

Your reasoning process:
1. First analyze the headline on its own
2. Search your memory for similar past articles
3. If the topic is complex or ambiguous, fetch related news
4. Decide final risk level based on all evidence
5. If risk >= 4, send an alert
6. Return your final structured analysis

Always return your final answer as a JSON object with these fields:
{
  "sentiment": "positive" | "negative" | "neutral",
  "sentiment_score": <float -1.0 to 1.0>,
  "impacted_sectors": [<list of sectors>],
  "stock_impact": "bullish" | "bearish" | "neutral",
  "risk_level": <integer 1-5>,
  "executive_summary": "<2-3 sentences>",
  "recommendations": "<actionable advice>",
  "agent_reasoning": "<explain what tools you used and why>",
  "historical_context": "<what past articles revealed>",
  "alert_triggered": true | false
}"""


def run_agent(headline: str) -> dict:
    messages = [{"role": "user", "content": f"Analyze this financial headline: {headline}"}]
    reasoning_steps = []
    max_iterations = 10
    iteration = 0

    print(f"\n🤖 Agent starting analysis for: {headline}\n")

    while iteration < max_iterations:
        iteration += 1
        print(f"--- Iteration {iteration} ---")

        response = client.messages.create(
            model="claude sonnet 4",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=TOOLS_DEFINITION,
            messages=messages
        )

        print(f"Stop reason: {response.stop_reason}")

        # Add assistant response to messages
        messages.append({"role": "assistant", "content": response.content})

        # If Claude is done reasoning
        if response.stop_reason == "end_turn":
            print("✅ Agent finished reasoning\n")
            for block in response.content:
                if hasattr(block, "text"):
                    raw = block.text.strip()
                    # Extract JSON from response
                    if "{" in raw:
                        start = raw.index("{")
                        end = raw.rindex("}") + 1
                        json_str = raw[start:end]
                        result = json.loads(json_str)
                        result["headline"] = headline
                        result["reasoning_steps"] = reasoning_steps
                        result["iterations"] = iteration
                        return result
            break

        # If Claude wants to use a tool
        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id

                    print(f"🔧 Claude calling tool: {tool_name}")
                    print(f"   Input: {json.dumps(tool_input)}")

                    reasoning_steps.append({
                        "iteration": iteration,
                        "tool_called": tool_name,
                        "input": tool_input
                    })

                    # Execute the tool
                    tool_fn = TOOL_MAP.get(tool_name)
                    if tool_fn:
                        tool_output = tool_fn(**tool_input)
                    else:
                        tool_output = {"error": f"Tool {tool_name} not found"}

                    print(f"   Result: {json.dumps(tool_output)[:200]}...")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": json.dumps(tool_output)
                    })

            # Feed tool results back to Claude
            messages.append({"role": "user", "content": tool_results})

    return {"error": "Agent did not complete within max iterations", "headline": headline}


if __name__ == "__main__":
    result = run_agent("India imposes emergency tariffs on Chinese steel imports amid dumping concerns")
    print("\n📊 FINAL AGENT OUTPUT:")
    print(json.dumps(result, indent=2))
