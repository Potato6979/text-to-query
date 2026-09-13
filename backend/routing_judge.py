import json
import re
from typing import Any

from openai import OpenAI

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from .llm_utils import call_chat_completion
from .routing_common import QUERY_TYPES, TOP_K_CANDIDATES
from .schema_plan_proposer import normalize_schema_plan_proposals


llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# 中文函数说明索引：
# - call_llm(prompt)：调用路由裁判使用的 LLM，保持 router 内部调用点集中。
# - judge_route_with_llm(question, evidence)：让 LLM 基于 evidence 判断 query_type；当前是辅助裁判，最终会再经过 guard 校准。


def call_llm(prompt: str) -> str:
    return call_chat_completion(llm, prompt, label="routing_llm")


def judge_route_with_llm(question: str, evidence: dict[str, Any]) -> dict[str, Any]:
    deterministic_plans = evidence.get("schema_plan_proposals", [])
    prompt = f"""You are the Route Judge inside a routing agent.
Do not decide from the raw question alone. Use the prepared evidence, compact schema context, and deterministic plan proposals.

Question:
{question}

Task analysis:
{json.dumps(evidence.get("task_analysis", {}), ensure_ascii=False)}

Intent signals:
{json.dumps(evidence["intent_signals"], ensure_ascii=False)}

Entity hints:
{json.dumps(evidence["entity_hints"], ensure_ascii=False)}

Modality scores:
{json.dumps(evidence["modality_scores"], ensure_ascii=False)}

Retrieval hints:
{json.dumps(evidence["retrieval_hints"], ensure_ascii=False)}

Schema plan context:
{json.dumps(evidence.get("schema_plan_context", {}), ensure_ascii=False)}

Deterministic schema plan proposals:
{json.dumps(deterministic_plans, ensure_ascii=False)}

The schema plan context has two layers:
- catalog: a global name-level catalog of indexed SQL databases/tables/columns and Cypher databases/node labels/relationship types.
- sql/cypher: a few top-ranked schema snippets for the current question.
It does not contain table rows, sample values, or full CREATE TABLE / graph DDL.

Tasks:
1. Use the provided Task analysis as the primary task-mode decision. Only change task_mode if the schema evidence strongly contradicts it.
2. Choose the best query_type.
3. Return top candidates in execution order.
4. Return confidence.
5. If uncertain, explain uncertainty_source.
6. Return up to 3 schema_plan_proposals ranked by likelihood. A proposal may refine the deterministic proposal, but should stay generic and evidence-based.

Return JSON only:
{{
  "task_mode": "single_step or multi_step_candidate",
  "query_type": "sql or cypher or mql or vector",
  "candidates": ["sql", "cypher"],
  "confidence": "high or medium or low",
  "reason": "one concise sentence",
  "uncertainty_source": "",
  "schema_plan_proposals": [
    {{
      "proposal_id": "P1",
      "rank": 1,
      "query_type": "sql",
      "query_strategy": "single_query",
      "query_shape": "direct_query",
      "schema_items": ["TableOrLabel"],
      "confidence": "medium",
      "source": "llm_schema_plan",
      "reason": "one concise sentence"
    }}
  ]
}}"""
    raw = call_llm(prompt)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("Route Judge did not return JSON.")
    result = json.loads(match.group())
    if result.get("query_type") not in QUERY_TYPES:
        raise ValueError("Route Judge returned an unsupported query_type.")
    candidates = [candidate for candidate in result.get("candidates", []) if candidate in QUERY_TYPES]
    if result["query_type"] not in candidates:
        candidates.insert(0, result["query_type"])
    result["candidates"] = candidates[:TOP_K_CANDIDATES]
    result["schema_plan_context"] = evidence.get("schema_plan_context", {})
    result["schema_plan_proposals"] = normalize_schema_plan_proposals(
        result.get("schema_plan_proposals"),
        deterministic_plans,
    )
    return result
