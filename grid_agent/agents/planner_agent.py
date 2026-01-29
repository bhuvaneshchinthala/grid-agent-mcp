import json
import ollama

class PlannerAgent:
    def __init__(self, model="mistral"):
        self.model = model
        self.last_reasoning = "No reasoning yet."

    def plan(self, violations):
        prompt = f"""
You are a power grid control agent.

Return ONLY valid JSON.
NO comments.
NO explanations.
NO trailing commas.
NO markdown.

Violations:
{json.dumps(violations, indent=2)}

Allowed actions:
- reduce_generation
- curtail_load
- switch_line

STRICT JSON FORMAT (example):
[
  {{
    "action_type": "reduce_generation",
    "target": 1,
    "value": -5
  }}
]
"""

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0,
                "num_predict": 256
            }
        )

        content = response["message"]["content"]

        # Save full LLM output for Streamlit UI
        self.last_reasoning = content

        # Extract and parse JSON safely
        try:
            start = content.find("[")
            end = content.rfind("]") + 1
            json_text = content[start:end]
            return json.loads(json_text)
        except Exception as e:
            print("PlannerAgent JSON parse error:", e)
            return []
