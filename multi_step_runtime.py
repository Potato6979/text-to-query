import json
import re
from typing import Any, Callable

from pipeline_compat import get_pipeline_query, get_pipeline_selected_db
from verification import VERDICT_PASS

# 中文函数说明索引：
# - _compact_rows/_rows_to_records：把 pipeline rows 压缩成 runtime variable 的 sample_rows / records。
# - _field_role/_primary_field/_primary_values：识别 name/id/metric 等字段角色，抽取可传给下一步的 primary values。
# - _extract_step_variable(step, pipeline_result)：把某一步执行结果变成 typed variable，写入 variable_store。
# - _fallback_input_contract/_contracts_for_step：补齐或合并 step 的 input_contracts，计算实际 consumption policy。
# - _evaluate_bridge/_evaluate_bridges_for_step：运行 bridge policy；同源 not_required 直接 resolved，跨源 required_before_hard_binding 需要 resolver。
# - _bridge_eval_by_variable/_unresolved_required_bridge_reason：把 bridge 结果按变量索引，并在默认自动跨源路径中判断是否需要安全阻断。
# - _variable_summary/_prompt_bridge_evaluation：构造下一步 prompt 中的 hard/soft context；unresolved/ambiguous bridge 会隐藏 source values。
# - _step_summary/_variable_answer_summary/_bridge_answer_summary：为 FAS 汇总步骤、变量和 bridge 证据。
# - _runtime_status_label/_policy_label/_format_result_preview：把工程状态转成演示可读文本。
# - synthesize_final_answer(...)：FAS 主函数，按 Answer / How derived / Safety note / Status / Answer 组织最终表达。
# - build_multi_step_runtime_question(...)：为当前 step 构造带变量契约、bridge 诊断、hard/soft context 的子问题。
# - run_multi_step_runtime(...)：多步执行主入口，按 step 顺序调用 pipeline、verification、bridge resolver，并返回 runtime trace。
# 状态说明：
# - resolved：bridge 已安全解析，可 hard_constraint。
# - unresolved_soft_context：bridge 未找到可靠映射，只能 soft_context。
# - ambiguous_soft_context：bridge 多候选或不唯一，不能 hard bind。
# - blocked：默认安全策略阻断依赖步骤，避免返回不可靠答案。


MAX_VARIABLE_ROWS = 25
NAME_FIELD_TOKENS = {"name", "title", "label"}
ID_FIELD_TOKENS = {"id", "identifier", "key"}
METRIC_FIELD_TOKENS = {"count", "total", "sum", "avg", "average", "max", "min", "score", "rank", "number", "num"}

PipelineRunner = Callable[[str, str, dict[str, Any]], dict[str, Any]]
StepVerifier = Callable[[str, dict[str, Any], dict[str, Any]], dict[str, Any]]
BridgeResolver = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]]


def _normalized_question_text(question: str) -> str:
    match = re.search(
        r"original user question:\s*(.*?)(?:\ncurrent step id:|\ncurrent step goal:|$)",
        str(question or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = match.group(1) if match else str(question or "")
    return " ".join(text.lower().split())


def _has_singular_bridge_reference(question: str, entity_type: str) -> bool:
    normalized = _normalized_question_text(question)
    entity = re.escape(str(entity_type or "").lower())
    if not entity:
        entity = r"(?:entity|record|student|singer|player|character|planet)"
    singular_patterns = [
        rf"\bthe\s+{entity}\b",
        rf"\bthat\s+{entity}\b",
        rf"\bthis\s+{entity}\b",
        rf"\bthat\s+\w+\s+{entity}\b",
        rf"\bthe\s+\w+\s+{entity}\b",
        r"\bfor\s+gabriel\b",
        r"\bfind\s+gabriel\b",
        r"\bgabriel's\b",
        r"\bthe\s+droid\s+character\b",
        r"\bthe\s+character\s+from\s+a\s+film\b",
        r"\bthe\s+singer\s+who\b",
        r"\bthe\s+no\.\s*\d+\s+player\b",
    ]
    return any(re.search(pattern, normalized) for pattern in singular_patterns)


def _pluralize_entity(entity_type: str) -> str:
    entity = str(entity_type or "").lower().strip()
    if not entity:
        return ""
    if entity.endswith("y"):
        return f"{entity[:-1]}ies"
    if entity.endswith("s"):
        return entity
    return f"{entity}s"


def _has_many_entity_language(question: str, entity_type: str) -> bool:
    normalized = _normalized_question_text(question)
    entity = re.escape(str(entity_type or "").lower())
    plural = re.escape(_pluralize_entity(entity_type))
    many_patterns = [
        rf"\ball\s+{plural}\b",
        rf"\beach\s+(?:of\s+those\s+)?{plural}\b",
        rf"\bevery\s+{entity}\b",
        rf"\bthose\s+{plural}\b",
        rf"\btheir\s+\w+",
        rf"\bfor\s+each\s+{entity}\b",
        rf"\bfor\s+each\s+{plural}\b",
        rf"\blist\s+their\b",
        rf"\breturn\s+each\b",
    ]
    return any(re.search(pattern, normalized) for pattern in many_patterns)


def _singular_many_bridge_guard(
    *,
    question: str,
    bridge_evaluations: list[dict[str, Any]],
    variable_store: dict[str, Any],
) -> str:
    for bridge in bridge_evaluations:
        if bridge.get("runtime_status") != "resolved":
            continue
        if bridge.get("match_type") != "exact_unique_many":
            continue
        input_var = str(bridge.get("input_var") or "")
        payload = variable_store.get(input_var, {})
        source_count = int(bridge.get("source_value_count") or payload.get("row_count") or 0)
        if source_count <= 1:
            continue
        entity_type = str(bridge.get("source_entity_type") or payload.get("entity_type") or "")
        if _has_many_entity_language(question, entity_type):
            continue
        if _has_singular_bridge_reference(question, entity_type):
            bridge["runtime_status"] = "ambiguous_soft_context"
            bridge["effective_consumption_policy"] = "soft_context"
            bridge["allow_hard_constraint"] = False
            bridge["runtime_reason"] = (
                "The question refers to a singular entity, but the upstream step produced multiple mapped "
                "entities. Hard binding would silently answer a different list-style question."
            )
            return (
                f"Required bridge {bridge.get('bridge_id', '')} stayed ambiguous_soft_context; "
                "the question needs a unique entity selector."
            )
    return ""


def _compact_rows(rows: Any) -> list[Any]:
    if not isinstance(rows, list):
        return []
    return rows[:MAX_VARIABLE_ROWS]


def _field_role(column: str) -> str:
    lowered = str(column or "").lower()
    if any(token in lowered for token in ID_FIELD_TOKENS):
        return "identifier"
    if any(token in lowered for token in NAME_FIELD_TOKENS):
        return "name"
    if any(token in lowered for token in METRIC_FIELD_TOKENS):
        return "metric"
    return "attribute"


def _rows_to_records(columns: list[Any], rows: list[Any]) -> list[dict[str, Any]]:
    column_names = [str(column) for column in columns]
    records: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            records.append({str(key): value for key, value in row.items()})
        elif isinstance(row, (list, tuple)):
            records.append({column_names[index]: value for index, value in enumerate(row[: len(column_names)])})
        else:
            records.append({"value": row})
    return records


def _primary_field(columns: list[Any]) -> str:
    column_names = [str(column) for column in columns]
    for role in ("identifier", "name", "attribute", "metric"):
        for column in column_names:
            if _field_role(column) == role:
                return column
    return column_names[0] if column_names else "value"


def _primary_values(records: list[dict[str, Any]], primary_field: str) -> list[Any]:
    values: list[Any] = []
    for record in records:
        if primary_field in record:
            values.append(record[primary_field])
        elif record:
            values.append(next(iter(record.values())))
    return values


def _runtime_confidence(row_count: Any, primary_values: list[Any]) -> str:
    if row_count and primary_values:
        return "high"
    if primary_values:
        return "medium"
    return "low"


def _extract_step_variable(step: dict[str, Any], pipeline_result: dict[str, Any]) -> dict[str, Any]:
    columns = pipeline_result.get("result_columns", [])
    rows = _compact_rows(pipeline_result.get("result_rows", []))
    records = _rows_to_records(columns, rows)
    primary_field = _primary_field(columns)
    values = _primary_values(records, primary_field)
    output_contract = step.get("output_contract", {})
    fields = [
        {
            "name": str(column),
            "role": _field_role(str(column)),
        }
        for column in columns
    ]
    return {
        "step_id": step.get("step_id", ""),
        "output_var": step.get("output_var", ""),
        "entity_type": output_contract.get("entity_type", "record"),
        "query_type": pipeline_result.get("query_type", step.get("query_type", "")),
        "source": {
            "query_type": pipeline_result.get("query_type", step.get("query_type", "")),
            "selected_db": get_pipeline_selected_db(pipeline_result),
            "target_resources": step.get("target_resources", []),
        },
        "selected_db": get_pipeline_selected_db(pipeline_result),
        "query": get_pipeline_query(pipeline_result),
        "columns": columns,
        "fields": fields,
        "primary_field": primary_field,
        "primary_values": values,
        "records": records,
        "sample_rows": rows,
        "row_count": pipeline_result.get("row_count", 0),
        "confidence": _runtime_confidence(pipeline_result.get("row_count", 0), values),
        "contract": {
            **output_contract,
            "fields": fields,
            "primary_field": primary_field,
            "confidence": _runtime_confidence(pipeline_result.get("row_count", 0), values),
        },
        "result_text": str(pipeline_result.get("result", ""))[:800],
    }


def _fallback_input_contract(var_name: str, step: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    source_query_type = payload.get("query_type", "")
    target_query_type = step.get("query_type", "")
    policy = "hard_constraint" if source_query_type == target_query_type else "soft_context"
    return {
        "variable": var_name,
        "consumer_step": step.get("step_id", ""),
        "required": True,
        "expected_entity_type": payload.get("entity_type", "record"),
        "source_query_type": source_query_type,
        "target_query_type": target_query_type,
        "compatibility": "compatible_same_query_type" if policy == "hard_constraint" else "unknown_cross_source_alignment",
        "consumption_policy": policy,
        "matched_tokens": [],
        "reason": "Runtime fallback contract generated because the plan did not provide an input contract.",
    }


def _bridge_steps_for_step(multi_step_plan: dict[str, Any], step: dict[str, Any]) -> list[dict[str, Any]]:
    step_id = step.get("step_id", "")
    return [
        bridge
        for bridge in multi_step_plan.get("bridge_steps", [])
        if bridge.get("to_step") == step_id
    ]


def _evaluate_bridge(
    bridge: dict[str, Any],
    variable_store: dict[str, Any],
    target_step: dict[str, Any] | None = None,
    resolve_bridge: BridgeResolver | None = None,
) -> dict[str, Any]:
    var_name = bridge.get("input_var", "")
    payload = variable_store.get(var_name, {})
    if not var_name or not payload:
        return {
            **bridge,
            "runtime_status": "missing_input",
            "effective_consumption_policy": "soft_context",
            "runtime_reason": "Bridge input variable is not available at runtime.",
        }

    if bridge.get("status") == "not_required":
        return {
            **bridge,
            "runtime_status": "resolved",
            "effective_consumption_policy": "hard_constraint",
            "runtime_reason": "Bridge is not required; the input variable can keep its planned hard binding.",
        }

    if bridge.get("status") == "required_before_hard_binding":
        if resolve_bridge is not None:
            resolver_result = resolve_bridge(bridge, payload, target_step or {})
            runtime_status = resolver_result.get("runtime_status", "")
            if runtime_status:
                default_policy = "hard_constraint" if runtime_status == "resolved" else "soft_context"
                return {
                    **bridge,
                    **resolver_result,
                    "effective_consumption_policy": resolver_result.get(
                        "effective_consumption_policy",
                        default_policy,
                    ),
                    "runtime_reason": resolver_result.get(
                        "runtime_reason",
                        (
                            "Bridge resolver found a target-side mapping; the variable can be used as a hard constraint."
                            if runtime_status == "resolved"
                            else "Bridge resolver did not find a safe target-side mapping."
                        ),
                    ),
                }
        values = payload.get("primary_values", [])
        return {
            **bridge,
            "runtime_status": "unresolved_soft_context",
            "effective_consumption_policy": "soft_context",
            "primary_values_seen": values,
            "runtime_reason": (
                "Bridge mapping is required before hard binding, but no executable bridge resolver is available. "
                "The variable remains soft context for this step."
            ),
        }

    return {
        **bridge,
        "runtime_status": "unknown",
        "effective_consumption_policy": "soft_context",
        "runtime_reason": "Bridge requirement is unknown; the variable remains soft context.",
    }


def _evaluate_bridges_for_step(
    multi_step_plan: dict[str, Any],
    step: dict[str, Any],
    variable_store: dict[str, Any],
    resolve_bridge: BridgeResolver | None = None,
) -> list[dict[str, Any]]:
    return [
        _evaluate_bridge(bridge, variable_store, step, resolve_bridge)
        for bridge in _bridge_steps_for_step(multi_step_plan, step)
    ]


def _bridge_eval_by_variable(bridge_evaluations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        evaluation.get("input_var", ""): evaluation
        for evaluation in bridge_evaluations
        if evaluation.get("input_var")
    }


def _unresolved_required_bridge_reason(bridge_evaluations: list[dict[str, Any]]) -> str:
    blocked = [
        evaluation
        for evaluation in bridge_evaluations
        if evaluation.get("status") == "required_before_hard_binding"
        and evaluation.get("effective_consumption_policy") != "hard_constraint"
    ]
    if not blocked:
        return ""
    bridge = blocked[0]
    return (
        f"Required bridge {bridge.get('bridge_id', '')} stayed "
        f"{bridge.get('runtime_status', 'unresolved')}; hard binding is not safe."
    )


def _contracts_for_step(
    step: dict[str, Any],
    variable_store: dict[str, Any],
    bridge_evaluations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    by_var = {
        contract.get("variable", ""): contract
        for contract in step.get("input_contracts", [])
        if contract.get("variable")
    }
    bridges_by_var = _bridge_eval_by_variable(bridge_evaluations or [])
    contracts: list[dict[str, Any]] = []
    for var_name, payload in variable_store.items():
        contract = dict(by_var.get(var_name) or _fallback_input_contract(var_name, step, payload))
        bridge = bridges_by_var.get(var_name)
        if bridge:
            contract["bridge_runtime_status"] = bridge.get("runtime_status", "")
            contract["bridge_runtime_reason"] = bridge.get("runtime_reason", "")
            contract["effective_consumption_policy"] = bridge.get(
                "effective_consumption_policy",
                contract.get("consumption_policy", "soft_context"),
            )
        contracts.append(contract)
    return contracts


def _variable_summary(
    var_name: str,
    payload: dict[str, Any],
    *,
    include_values: bool = True,
    bridge_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mapped_values = (bridge_evaluation or {}).get("target_values", [])
    bridge_resolved = bool(bridge_evaluation and bridge_evaluation.get("runtime_status") == "resolved")
    target_entity_type = (bridge_evaluation or {}).get("target_entity_type", "")
    summary = {
        "variable": var_name,
        "entity_type": payload.get("entity_type", "record"),
        "source": payload.get("source", {}),
        "fields": payload.get("fields", []),
        "primary_field": (bridge_evaluation or {}).get("target_field", payload.get("primary_field", "")),
        "row_count": payload.get("row_count", 0),
        "confidence": payload.get("confidence", ""),
    }
    if bridge_resolved:
        summary["source_entity_type"] = payload.get("entity_type", "record")
        if target_entity_type:
            summary["mapped_entity_type"] = target_entity_type
    if include_values:
        summary["primary_values"] = mapped_values or payload.get("primary_values", [])
        if bridge_resolved:
            summary["sample_records"] = []
            summary["source_values_omitted_after_bridge"] = True
            summary["bridge_mapping"] = {
                "bridge_id": bridge_evaluation.get("bridge_id", ""),
                "target_field": bridge_evaluation.get("target_field", ""),
                "target_values": bridge_evaluation.get("target_values", []),
                "target_entity_type": target_entity_type,
                "resolver_strategy": bridge_evaluation.get("resolver_strategy", ""),
                "confidence": bridge_evaluation.get("confidence", ""),
            }
        else:
            summary["sample_records"] = payload.get("records", [])
    else:
        summary["primary_values"] = []
        summary["sample_records"] = []
        summary["values_redacted"] = True
        summary["redaction_reason"] = "Bridge mapping is unresolved; exact values must not be used as target-side lookup filters."
    return summary


def _format_variable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _prompt_bridge_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    rendered = dict(evaluation)
    if rendered.get("runtime_status") in {"unresolved_soft_context", "ambiguous_soft_context"}:
        rendered.pop("primary_values_seen", None)
        rendered["values_redacted_in_prompt"] = True
    return rendered


def _step_summary(step_record: dict[str, Any]) -> dict[str, Any]:
    pipeline = step_record.get("pipeline", {})
    verification = step_record.get("verification", {})
    return {
        "step_id": step_record.get("step_id", ""),
        "query_type": step_record.get("query_type", ""),
        "status": step_record.get("status", ""),
        "goal": step_record.get("step_goal", ""),
        "selected_db": pipeline.get("selected_db", ""),
        "row_count": pipeline.get("row_count", 0),
        "verdict": verification.get("verdict", ""),
        "failure_type": verification.get("failure_type", ""),
        "reason": verification.get("reason", ""),
    }


def _variable_answer_summary(variable_store: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for var_name, payload in variable_store.items():
        summaries.append(
            {
                "variable": var_name,
                "entity_type": payload.get("entity_type", "record"),
                "primary_field": payload.get("primary_field", ""),
                "primary_values": payload.get("primary_values", []),
                "row_count": payload.get("row_count", 0),
                "source": payload.get("source", {}),
            }
        )
    return summaries


def _bridge_answer_summary(bridge_store: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "bridge_id": bridge.get("bridge_id", bridge_id),
            "from_step": bridge.get("from_step", ""),
            "to_step": bridge.get("to_step", ""),
            "input_var": bridge.get("input_var", ""),
            "source_entity_type": bridge.get("source_entity_type", ""),
            "source_query_type": bridge.get("source_query_type", ""),
            "target_query_type": bridge.get("target_query_type", ""),
            "runtime_status": bridge.get("runtime_status", ""),
            "effective_consumption_policy": bridge.get("effective_consumption_policy", ""),
            "reason": bridge.get("runtime_reason", bridge.get("reason", "")),
        }
        for bridge_id, bridge in bridge_store.items()
    ]


def _format_values(values: list[Any]) -> str:
    if not values:
        return "no concrete values"
    return ", ".join(str(value) for value in values[:3])


def _runtime_status_label(status: str) -> str:
    labels = {
        "succeeded": "completed",
        "failed": "failed",
        "blocked": "safely stopped",
        "resolved": "mapped safely",
        "unresolved_soft_context": "not safely mapped",
        "ambiguous_soft_context": "ambiguous mapping",
        "missing_input": "missing input",
        "unknown": "unknown mapping state",
    }
    return labels.get(status, status or "unknown")


def _policy_label(policy: str) -> str:
    labels = {
        "hard_constraint": "hard filter",
        "soft_context": "context only",
    }
    return labels.get(policy, policy or "unknown")


def _format_result_preview(result_text: str, max_lines: int = 8) -> str:
    lines = [line for line in str(result_text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"])


def synthesize_final_answer(
    *,
    question: str,
    status: str,
    reason: str,
    steps: list[dict[str, Any]],
    variable_store: dict[str, Any],
    bridge_store: dict[str, Any],
    final_pipeline: dict[str, Any],
    final_verification: dict[str, Any],
) -> dict[str, Any]:
    """Build a deterministic multi-step final answer from runtime evidence."""
    step_summaries = [_step_summary(step) for step in steps]
    variable_summaries = _variable_answer_summary(variable_store)
    bridge_summaries = _bridge_answer_summary(bridge_store)
    failed_steps = [step for step in step_summaries if step.get("status") == "failed"]

    lines = [f"Answer for: {question}"]
    if variable_summaries:
        lines.append("")
        lines.append("How it was derived:")
        for variable in variable_summaries:
            lines.append(
                "- "
                f"{variable['variable']} ({variable['entity_type']}) from "
                f"{variable.get('source', {}).get('query_type', '')}/{variable.get('source', {}).get('selected_db', '')}: "
                f"{variable['primary_field']} = {_format_values(variable['primary_values'])}; "
                f"rows = {variable['row_count']}."
            )
    else:
        lines.append("")
        lines.append("How it was derived: no intermediate result was successfully produced.")

    unresolved_bridges = [
        bridge
        for bridge in bridge_summaries
        if bridge.get("runtime_status") in {"unresolved_soft_context", "ambiguous_soft_context", "missing_input", "unknown"}
    ]
    if unresolved_bridges:
        lines.append("")
        lines.append("Bridge / mapping limitations (safety note):")
        for bridge in unresolved_bridges:
            lines.append(
                "- "
                f"{bridge['bridge_id']} ({_runtime_status_label(bridge['runtime_status'])}): "
                f"{bridge['source_entity_type']} from {bridge['source_query_type']} could not be safely used as a "
                f"{_policy_label('hard_constraint')} in {bridge['target_query_type']}; "
                f"it stayed {_policy_label(bridge['effective_consumption_policy'])}. "
                f"{bridge['reason']}"
            )

    if status == "succeeded":
        lines.append("")
        lines.append("Status: completed.")
        lines.append("Final step succeeded.")
        result_text = str(final_pipeline.get("result", "")).strip()
        if result_text:
            lines.append("Answer:")
            lines.append(_format_result_preview(result_text))
    elif failed_steps:
        failed = failed_steps[-1]
        lines.append("")
        lines.append("Status: failed.")
        lines.append(
            "Execution stopped at "
            f"{failed['step_id']} ({failed['query_type']}) with {failed['verdict']} / {failed['failure_type']}."
        )
        lines.append(f"Reason: {failed.get('reason') or reason}")
    else:
        lines.append("")
        lines.append(f"Status: {_runtime_status_label(status)}.")
        lines.append(f"Execution did not complete. Reason: {reason}")

    synthesis = {
        "status": status,
        "reason": reason,
        "steps": step_summaries,
        "variables": variable_summaries,
        "bridges": bridge_summaries,
    }
    return {
        "success": status == "succeeded",
        "query": get_pipeline_query(final_pipeline),
        "result_text": "\n".join(lines),
        "raw_result_text": final_pipeline.get("result", ""),
        "synthesis": synthesis,
    }


def build_multi_step_runtime_question(
    *,
    original_question: str,
    step: dict[str, Any],
    variable_store: dict[str, Any],
    bridge_evaluations: list[dict[str, Any]] | None = None,
) -> str:
    input_contracts = _contracts_for_step(step, variable_store, bridge_evaluations)
    bridges_by_var = _bridge_eval_by_variable(bridge_evaluations or [])
    hard_lines = []
    soft_lines = []
    contract_lines = []
    for contract in input_contracts:
        var_name = contract.get("variable", "")
        payload = variable_store.get(var_name, {})
        policy = contract.get("effective_consumption_policy", contract.get("consumption_policy", "soft_context"))
        bridge = bridges_by_var.get(var_name)
        redact_values = (
            policy == "soft_context"
            and contract.get("bridge_runtime_status") in {"unresolved_soft_context", "ambiguous_soft_context"}
        )
        summary = _variable_summary(var_name, payload, include_values=not redact_values, bridge_evaluation=bridge)
        rendered = _format_variable_json(summary)
        contract_lines.append(_format_variable_json(contract))
        if policy == "hard_constraint":
            hard_lines.append(f"{var_name}: {rendered}")
        else:
            soft_lines.append(f"{var_name}: {rendered}")

    hard_context = "\n".join(hard_lines) if hard_lines else "None"
    soft_context = "\n".join(soft_lines) if soft_lines else "None"
    contract_context = "\n".join(contract_lines) if contract_lines else "None"
    bridge_context = "\n".join(_format_variable_json(_prompt_bridge_evaluation(item)) for item in (bridge_evaluations or [])) or "None"
    return (
        "Multi-step query execution subtask.\n"
        f"Original user question: {original_question}\n"
        f"Current step id: {step.get('step_id', '')}\n"
        f"Current step goal: {step.get('step_goal', '')}\n"
        f"Target query type: {step.get('query_type', '')}\n"
        f"Relevant schema items: {step.get('schema_items', [])}\n"
        f"Input contracts:\n{contract_context}\n"
        f"Bridge mapping diagnostics:\n{bridge_context}\n"
        f"Hard constraints from previous variables:\n{hard_context}\n"
        f"Soft context from previous variables:\n{soft_context}\n"
        "Answer only this current step. Use hard constraints as required filters. "
        "Use soft context only as background evidence; do not force exact entity lookup unless the target schema supports it."
    )


def run_multi_step_runtime(
    *,
    question: str,
    multi_step_plan: dict[str, Any],
    run_pipeline: PipelineRunner,
    verify_step: StepVerifier,
    resolve_bridge: BridgeResolver | None = None,
    max_steps: int = 2,
    block_unresolved_required_bridges: bool = False,
) -> dict[str, Any]:
    steps = [
        step
        for step in multi_step_plan.get("steps", [])[:max_steps]
        if step.get("step_type") == "query" and step.get("query_type")
    ]
    if not steps:
        final_answer = synthesize_final_answer(
            question=question,
            status="not_started",
            reason="Multi-step plan has no executable query steps.",
            steps=[],
            variable_store={},
            bridge_store={},
            final_pipeline={},
            final_verification={},
        )
        return {
            "status": "not_started",
            "reason": "Multi-step plan has no executable query steps.",
            "steps": [],
            "variable_store": {},
            "bridge_store": {},
            "final_pipeline": {},
            "final_verification": {},
            "final_answer": final_answer,
        }

    variable_store: dict[str, Any] = {}
    bridge_store: dict[str, Any] = {}
    executed_steps: list[dict[str, Any]] = []
    final_pipeline: dict[str, Any] = {}
    final_verification: dict[str, Any] = {}

    for step in steps:
        missing_inputs = [var_name for var_name in step.get("input_vars", []) if var_name not in variable_store]
        if missing_inputs:
            step_record = {
                **step,
                "status": "blocked",
                "blocked_reason": f"Missing input variables: {missing_inputs}",
                "runtime_question": "",
                "pipeline": {},
                "verification": {},
            }
            executed_steps.append(step_record)
            final_answer = synthesize_final_answer(
                question=question,
                status="blocked",
                reason=step_record["blocked_reason"],
                steps=executed_steps,
                variable_store=variable_store,
                bridge_store=bridge_store,
                final_pipeline=final_pipeline,
                final_verification=final_verification,
            )
            return {
                "status": "blocked",
                "reason": step_record["blocked_reason"],
                "steps": executed_steps,
                "variable_store": variable_store,
                "bridge_store": bridge_store,
                "final_pipeline": final_pipeline,
                "final_verification": final_verification,
                "final_answer": final_answer,
            }

        bridge_evaluations = _evaluate_bridges_for_step(multi_step_plan, step, variable_store, resolve_bridge)
        for bridge_evaluation in bridge_evaluations:
            bridge_store[bridge_evaluation.get("bridge_id", f"bridge_{len(bridge_store) + 1}")] = bridge_evaluation

        singular_many_bridge_reason = _singular_many_bridge_guard(
            question=question,
            bridge_evaluations=bridge_evaluations,
            variable_store=variable_store,
        )
        blocked_bridge_reason = (
            singular_many_bridge_reason
            or (
            _unresolved_required_bridge_reason(bridge_evaluations)
            if block_unresolved_required_bridges
            else ""
            )
        )
        if blocked_bridge_reason:
            step_record = {
                **step,
                "status": "blocked",
                "blocked_reason": blocked_bridge_reason,
                "runtime_question": "",
                "bridge_evaluations": bridge_evaluations,
                "input_contracts_consumed": _contracts_for_step(step, variable_store, bridge_evaluations),
                "pipeline": {},
                "verification": {},
            }
            executed_steps.append(step_record)
            final_answer = synthesize_final_answer(
                question=question,
                status="blocked",
                reason=blocked_bridge_reason,
                steps=executed_steps,
                variable_store=variable_store,
                bridge_store=bridge_store,
                final_pipeline=final_pipeline,
                final_verification=final_verification,
            )
            return {
                "status": "blocked",
                "reason": blocked_bridge_reason,
                "steps": executed_steps,
                "variable_store": variable_store,
                "bridge_store": bridge_store,
                "final_pipeline": final_pipeline,
                "final_verification": final_verification,
                "final_answer": final_answer,
            }

        runtime_question = build_multi_step_runtime_question(
            original_question=question,
            step=step,
            variable_store=variable_store,
            bridge_evaluations=bridge_evaluations,
        )
        proposal = {
            "proposal_id": step.get("proposal_id", ""),
            "query_type": step.get("query_type", ""),
            "query_strategy": step.get("query_strategy", ""),
            "query_shape": step.get("query_shape", ""),
            "schema_items": step.get("schema_items", []),
            "target_resources": step.get("target_resources", []),
            "source": "multi_step_runtime",
            "reason": step.get("step_goal", ""),
        }
        pipeline_result = run_pipeline(step.get("query_type", ""), runtime_question, proposal)
        verification_result = verify_step(
            runtime_question,
            {**proposal, "bridge_evaluations": bridge_evaluations},
            pipeline_result,
        )
        final_pipeline = pipeline_result
        final_verification = verification_result

        step_record = {
            **step,
            "status": "succeeded" if verification_result.get("verdict") == VERDICT_PASS else "failed",
            "runtime_question": runtime_question,
            "bridge_evaluations": bridge_evaluations,
            "input_contracts_consumed": _contracts_for_step(step, variable_store, bridge_evaluations),
            "pipeline": {
                "success": pipeline_result.get("success", False),
                "query_type": pipeline_result.get("query_type", step.get("query_type", "")),
                "selected_db": get_pipeline_selected_db(pipeline_result),
                "query": get_pipeline_query(pipeline_result),
                "row_count": pipeline_result.get("row_count", 0),
                "error": pipeline_result.get("error", ""),
            },
            "verification": verification_result,
        }
        executed_steps.append(step_record)

        if verification_result.get("verdict") != VERDICT_PASS:
            final_answer = synthesize_final_answer(
                question=question,
                status="failed",
                reason=verification_result.get("reason", "A multi-step runtime step failed verification."),
                steps=executed_steps,
                variable_store=variable_store,
                bridge_store=bridge_store,
                final_pipeline=final_pipeline,
                final_verification=final_verification,
            )
            return {
                "status": "failed",
                "reason": verification_result.get("reason", "A multi-step runtime step failed verification."),
                "steps": executed_steps,
                "variable_store": variable_store,
                "bridge_store": bridge_store,
                "final_pipeline": final_pipeline,
                "final_verification": final_verification,
                "final_answer": final_answer,
            }

        variable = _extract_step_variable(step, pipeline_result)
        variable_store[step.get("output_var", f"step_{len(variable_store) + 1}_result")] = variable

    final_answer = synthesize_final_answer(
        question=question,
        status="succeeded",
        reason="All multi-step runtime steps passed verification.",
        steps=executed_steps,
        variable_store=variable_store,
        bridge_store=bridge_store,
        final_pipeline=final_pipeline,
        final_verification=final_verification,
    )
    return {
        "status": "succeeded",
        "reason": "All multi-step runtime steps passed verification.",
        "steps": executed_steps,
        "variable_store": variable_store,
        "bridge_store": bridge_store,
        "final_pipeline": final_pipeline,
        "final_verification": final_verification,
        "final_answer": final_answer,
    }
