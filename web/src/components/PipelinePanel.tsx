import type { DevQueryResponse } from "../types/api";
import { Icon } from "./Icon";
import { StepCard } from "./StepCard";

type Props = {
  response?: DevQueryResponse;
};

type UnknownRecord = Record<string, unknown>;

// 中文函数说明：
// - joinLines：把 StepCard 中的多段 trace 文本拼成可读内容。
// - asRecord/asArray/asText：把后端返回的 unknown JSON 安全转换成前端可渲染结构，避免 trace 字段缺失时报错。
// - explain：把后端工程状态码翻译成演示可读标签，例如 resolved -> safely mapped。
// - statusClass：根据状态类型选择徽标颜色，绿色表示完成/安全映射，黄色表示失败/阻断/未映射。
// - Badge/ExplainBadge：渲染原始状态或解释后的状态。
// - MultiStepTrace：展示 route_plan、runtime_decision、runtime steps、bridge safety 和 variable_store。
// - PipelinePanel：开发者模式右侧总 trace 面板，包含 Routing、Retrieval、Generation、Execution、Verification、Coordinator。

function joinLines(lines: Array<string | undefined | null | false>) {
  return lines.filter(Boolean).join("\n");
}

function asRecord(value: unknown): UnknownRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as UnknownRecord) : {};
}

function asArray(value: unknown): UnknownRecord[] {
  return Array.isArray(value) ? value.map(asRecord) : [];
}

function asText(value: unknown, fallback = "n/a") {
  if (value === null || value === undefined || value === "") return fallback;
  if (Array.isArray(value)) return value.join(", ") || fallback;
  return String(value);
}

function explain(value: unknown) {
  const text = asText(value, "");
  const labels: Record<string, string> = {
    auto_policy: "自动策略",
    explicit_override: "手动覆盖",
    single_step_runtime: "单步执行",
    runtime_not_executed: "未进入多步执行",
    auto_runtime_experiment_candidate: "自动多步候选",
    cross_source_runtime_auto_enabled_for_bridge_preflight: "跨源桥接预检",
    bridge_resolver_auto_enabled_with_high_confidence_gate: "桥接解析器高置信门控",
    explicit_runtime_only: "跨源预检",
    hard_constraint: "强约束",
    soft_context: "弱上下文",
    not_required: "无需桥接",
    planned_not_executed: "已规划待执行",
    resolved: "安全映射",
    unresolved_soft_context: "未安全映射",
    ambiguous_soft_context: "映射歧义",
    missing_input: "缺少输入",
    blocked: "安全阻断",
    answered: "已回答",
    succeeded: "已完成",
    failed: "失败",
    fail_terminal: "终止失败",
    pass: "通过",
    local_repair: "局部修复",
    regenerate: "重新生成",
    reroute: "重新路由",
  };
  return labels[text] ? `${labels[text]} (${text})` : text || "n/a";
}

function statusClass(status: string) {
  if (["succeeded", "verified", "resolved", "pass", "ready", "answered"].includes(status.toLowerCase())) {
    return "border-[#cadbd0] bg-[#e8f0eb] text-brand-pine";
  }
  if (["failed", "blocked", "unresolved_soft_context", "ambiguous_soft_context", "missing_input", "fail_terminal"].includes(status.toLowerCase())) {
    return "border-[#d8c490] bg-[#f4eddc] text-[#765f2f]";
  }
  return "border-brand-line bg-brand-paper text-brand-soft";
}

function Badge({ value }: { value: unknown }) {
  const text = asText(value);
  return <span className={`rounded-full border px-2 py-1 text-[11px] font-semibold ${statusClass(text)}`}>{text}</span>;
}

function ExplainBadge({ value }: { value: unknown }) {
  const text = explain(value);
  return <span className={`rounded-full border px-2 py-1 text-[11px] font-semibold ${statusClass(asText(value))}`}>{text}</span>;
}

function MultiStepTrace({ response }: { response: DevQueryResponse }) {
  const routePlan = response.pipeline.route.route_plan;
  const plan = asRecord(routePlan?.multi_step_plan ?? response.pipeline.route.multi_step_plan);
  const runtime = asRecord(response.pipeline.route.multi_step_runtime);
  const policy = asRecord(routePlan?.multi_step_runtime_policy);
  const decision = asRecord(response.pipeline.route.runtime_decision ?? response.runtime_decision);
  const planSteps = asArray(plan.steps).filter((step) => asText(step.step_type, "query") === "query");
  const bridgeSteps = asArray(plan.bridge_steps);
  const runtimeSteps = asArray(runtime.steps);
  const variables = asRecord(runtime.variable_store);
  const bridges = Object.values(asRecord(runtime.bridge_store)).map(asRecord);
  const hasTrace = planSteps.length > 0 || runtimeSteps.length > 0 || bridgeSteps.length > 0 || Object.keys(policy).length > 0;

  if (!hasTrace) return null;

  return (
    <section className="space-y-3 overflow-y-auto rounded-lg border border-brand-line bg-white/85 p-4 pr-3 shadow-sm lg:max-h-[420px]">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="flex items-center gap-2 text-sm font-bold text-brand-ink">
            <Icon name="link" className="h-4 w-4" />
            多步执行与桥接解析
          </p>
          <p className="mt-1 text-xs text-brand-muted">
            计划 {explain(plan.status || "not_required")} | 执行 {explain(runtime.status || "runtime_not_executed")}
          </p>
        </div>
        {policy.recommended_mode ? <ExplainBadge value={policy.recommended_mode} /> : null}
      </div>

      {Object.keys(policy).length ? (
        <div className="grid gap-2 rounded-lg border border-brand-line bg-brand-paper p-3 text-xs text-brand-soft">
          <p className="font-semibold text-brand-ink">多步执行策略</p>
          <div className="flex flex-wrap gap-2">
            <Badge value={`default=${asText(policy.default_enable, "false")}`} />
            <Badge value={`risk=${asText(policy.risk_level, "n/a")}`} />
          </div>
          {Array.isArray(policy.reasons) && policy.reasons.length ? (
            <p>{policy.reasons.map(String).join(" · ")}</p>
          ) : null}
        </div>
      ) : null}

      {Object.keys(decision).length ? (
        <div className="grid gap-2 rounded-lg border border-brand-line bg-[#eef3f5] p-3 text-xs text-brand-soft">
          <p className="font-semibold text-brand-ink">实际执行决策</p>
          <div className="flex flex-wrap gap-2">
            <Badge value={`runtime=${asText(decision.effective_multi_step_runtime, "false")}`} />
            <Badge value={`bridge=${asText(decision.effective_bridge_resolver, "false")}`} />
            <ExplainBadge value={decision.decision_source ?? "source=n/a"} />
          </div>
          <p>多步执行: {explain(decision.decision_reason)}</p>
          {decision.bridge_decision_reason ? <p>桥接解析: {explain(decision.bridge_decision_reason)}</p> : null}
        </div>
      ) : null}

      {planSteps.length ? (
        <div className="space-y-2">
          <p className="text-xs font-semibold text-brand-ink">规划与执行步骤</p>
          {planSteps.map((step, index) => {
            const output = asRecord(step.output_contract);
            const targets = asArray(step.target_resources);
            const runtimeStep = runtimeSteps.find((item) => item.step_id === step.step_id) ?? {};
            const pipeline = asRecord(runtimeStep.pipeline);
            const verification = asRecord(runtimeStep.verification);
            return (
              <div className="rounded-lg border border-brand-line bg-white p-3" key={`${asText(step.step_id)}-${index}`}>
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="text-xs font-semibold text-brand-ink">
                    {asText(step.step_id, `MS${index + 1}`)} · {asText(step.query_type)} · {asText(targets[0]?.resource_id)}
                  </p>
                  <Badge value={runtimeStep.status ?? step.status ?? "planned"} />
                </div>
                <p className="mt-2 text-xs text-brand-soft">{asText(step.step_goal ?? step.goal, "No step goal recorded.")}</p>
                <div className="mt-2 grid gap-1 text-[11px] text-brand-muted">
                  <span>输出变量: {asText(output.variable ?? step.output_var)} / {asText(output.entity_type)}</span>
                  <span>依赖变量: {asText(step.depends_on, "none")}</span>
                  {pipeline.row_count !== undefined ? <span>结果行数: {asText(pipeline.row_count)} · 数据库: {asText(pipeline.selected_db)}</span> : null}
                  {verification.verdict ? <span>验证: {asText(verification.verdict)} · {asText(verification.failure_type, "none")}</span> : null}
                  {runtimeStep.blocked_reason ? <span>安全终止: {asText(runtimeStep.blocked_reason)}</span> : null}
                </div>
              </div>
            );
          })}
        </div>
      ) : null}

      {bridgeSteps.length || bridges.length ? (
        <div className="space-y-2">
          <p className="text-xs font-semibold text-brand-ink">桥接安全</p>
          {[...bridgeSteps, ...bridges].map((bridge, index) => (
            <div className="rounded-lg border border-brand-line bg-brand-paper p-3 text-xs text-brand-soft" key={`bridge-${index}`}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="font-semibold text-brand-ink">
                  {asText(bridge.bridge_id, `BR${index + 1}`)} · {asText(bridge.from_step)} → {asText(bridge.to_step)}
                </span>
                <ExplainBadge value={bridge.runtime_status ?? bridge.status ?? "planned"} />
              </div>
              <p className="mt-2">
                {asText(bridge.source_entity_type)} → {asText(bridge.target_entity_type)} · {explain(bridge.effective_consumption_policy ?? bridge.consumption_policy)}
              </p>
              {bridge.match_type ? <p className="mt-1 text-brand-muted">匹配类型: {asText(bridge.match_type)} · 候选数: {asText(bridge.candidate_count, "0")}</p> : null}
              {bridge.runtime_reason ? <p className="mt-1 text-brand-muted">原因: {asText(bridge.runtime_reason)}</p> : null}
            </div>
          ))}
        </div>
      ) : null}

      {Object.keys(variables).length ? (
        <div className="rounded-lg border border-brand-line bg-brand-paper p-3 text-xs text-brand-soft">
          <p className="font-semibold text-brand-ink">中间变量</p>
          <div className="mt-2 space-y-1">
            {Object.entries(variables).map(([name, payload]) => {
              const variable = asRecord(payload);
              return (
                <p key={name}>
                  {name}: {asText(variable.entity_type)} · {asText(variable.primary_field)} = {asText(variable.primary_values)}
                </p>
              );
            })}
          </div>
        </div>
      ) : null}
    </section>
  );
}

export function PipelinePanel({ response }: Props) {
  if (!response) {
    return (
      <aside className="mt-5 flex flex-col rounded-lg border border-white/80 bg-white/75 p-5 shadow-panel backdrop-blur">
        <div className="flex items-center justify-between gap-3 border-b border-brand-line pb-4">
          <p className="flex items-center gap-2 text-sm font-bold uppercase tracking-[0.08em] text-brand-ink">
            <Icon name="route" className="h-4 w-4" />
            Execution
          </p>
          <span className="inline-flex items-center gap-2 text-xs font-bold text-brand-pine">
            <span className="h-2 w-2 rounded-full bg-brand-pine shadow-[0_0_0_5px_rgba(37,72,62,0.10)]" />
            Ready
          </span>
        </div>
        <div className="mt-5 space-y-5">
          {([
            ["Routing Agent", "识别 SQL、Cypher 或跨源执行候选", "route"],
            ["Task Mode Analyzer", "判断单步、嵌套、多步或跨源模式", "layers"],
            ["Bridge Resolver", "验证实体映射是否唯一、完整、连续", "link"],
            ["Verification + FAS", "检查结果并生成最终回答", "shield"],
          ] as const).map(([title, text, icon]) => (
            <div className="grid grid-cols-[28px_1fr] gap-3" key={title}>
              <div className="grid h-7 w-7 place-items-center rounded-full border border-brand-line bg-white text-brand-muted">
                <Icon name={icon} className="h-4 w-4" />
              </div>
              <div>
                <p className="text-sm font-bold text-brand-ink">{title}</p>
                <p className="mt-1 text-xs leading-5 text-brand-muted">{text}</p>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-auto space-y-3 border-t border-brand-line pt-4 text-xs">
          <div className="flex items-center justify-between gap-3 text-brand-muted"><span>Runtime Scope</span><strong className="text-brand-ink">SQL + Cypher</strong></div>
          <div className="flex items-center justify-between gap-3 text-brand-muted"><span>Safety Policy</span><strong className="text-brand-ink">High-confidence</strong></div>
          <div className="flex items-center justify-between gap-3 text-brand-muted"><span>Trace</span><strong className="text-brand-ink">Waiting</strong></div>
        </div>
      </aside>
    );
  }

  const { pipeline, timing } = response;

  const routeContent = joinLines([
    `Reason: ${pipeline.route.reason || "N/A"}`,
    `Task Mode: ${pipeline.route.task_mode || "single_step"}`,
    `Keep Backup Route: ${pipeline.route.should_keep_backup_route ? "yes" : "no"}`,
    pipeline.route.uncertainty_source ? `Uncertainty: ${pipeline.route.uncertainty_source}` : "",
    `Time: ${timing.route_ms}ms`,
    pipeline.route.candidates?.length ? "" : "",
    pipeline.route.candidates?.length ? "Candidates:" : "",
    ...(pipeline.route.candidates?.map(
      (candidate, index) =>
        `${index + 1}. ${candidate.query_type} | score=${candidate.score ?? "n/a"} | confidence=${candidate.confidence ?? "n/a"}${candidate.reason ? ` | ${candidate.reason}` : ""}`,
    ) ?? []),
    pipeline.route.signals ? "" : "",
    pipeline.route.signals ? "Signals:" : "",
    pipeline.route.signals ? JSON.stringify(pipeline.route.signals, null, 2) : "",
    pipeline.route.route_plan ? "" : "",
    pipeline.route.route_plan ? "Route Plan:" : "",
    pipeline.route.route_plan ? JSON.stringify(pipeline.route.route_plan, null, 2) : "",
  ]);

  const verificationContent = pipeline.verification
    ? joinLines([
        `Verdict: ${pipeline.verification.verdict}`,
        `Suggested Action: ${pipeline.verification.suggested_action || "N/A"}`,
        `Failure Type: ${pipeline.verification.failure_type || "none"}`,
        `Confidence: ${pipeline.verification.confidence || "N/A"}`,
        `Reason: ${pipeline.verification.reason || "N/A"}`,
        pipeline.verification.contract ? "" : "",
        pipeline.verification.contract ? "Contract:" : "",
        pipeline.verification.contract ? JSON.stringify(pipeline.verification.contract, null, 2) : "",
        pipeline.verification.generation_contract_check ? "" : "",
        pipeline.verification.generation_contract_check ? "Generation Contract Check:" : "",
        pipeline.verification.generation_contract_check
          ? JSON.stringify(pipeline.verification.generation_contract_check, null, 2)
          : "",
        pipeline.verification.checks ? "" : "",
        pipeline.verification.checks ? "Checks:" : "",
        pipeline.verification.checks ? JSON.stringify(pipeline.verification.checks, null, 2) : "",
      ])
    : "No verification data";

  const coordinatorContent = pipeline.coordinator
    ? joinLines([
        `Task ID: ${pipeline.coordinator.task_id}`,
        `Mode: ${pipeline.coordinator.mode}`,
        `Status: ${pipeline.coordinator.status}`,
        `Active Step: ${pipeline.coordinator.active_step_id}`,
        `Loop Count: ${pipeline.coordinator.loop_count}/${pipeline.coordinator.max_loops}`,
        `Step Retry Count: ${pipeline.coordinator.step_retry_count}`,
        `Reroute Count: ${pipeline.coordinator.reroute_count}`,
        pipeline.coordinator.decision_history?.length ? "" : "",
        pipeline.coordinator.decision_history?.length ? "Decision History:" : "",
        pipeline.coordinator.decision_history?.length
          ? JSON.stringify(pipeline.coordinator.decision_history, null, 2)
          : "",
      ])
    : "No coordinator data";

  const generationContent = joinLines([
    pipeline.generation.cot_reasoning || pipeline.generation.query,
    pipeline.generation.generation_contract ? "" : "",
    pipeline.generation.generation_contract ? "Generation Contract:" : "",
    pipeline.generation.generation_contract ? JSON.stringify(pipeline.generation.generation_contract, null, 2) : "",
    pipeline.generation.generation_feedback ? "" : "",
    pipeline.generation.generation_feedback ? "Generation Feedback:" : "",
    pipeline.generation.generation_feedback ? JSON.stringify(pipeline.generation.generation_feedback, null, 2) : "",
    pipeline.generation.task_analysis ? "" : "",
    pipeline.generation.task_analysis ? "Task Analysis:" : "",
    pipeline.generation.task_analysis ? JSON.stringify(pipeline.generation.task_analysis, null, 2) : "",
  ]);

  return (
    <aside className="mt-5 flex min-w-0 flex-col rounded-lg border border-white/80 bg-white/75 p-5 shadow-panel backdrop-blur">
      <div className="shrink-0">
        <h2 className="flex items-center gap-2 text-lg font-semibold text-brand-ink">
          <Icon name="route" className="h-5 w-5" />
          执行 Trace
        </h2>
        <p className="mt-1 text-xs text-brand-muted">总耗时 {timing.total_ms}ms · SQL/Cypher 双源范围</p>
      </div>
      <div className="mt-4 space-y-4">
        <MultiStepTrace response={response} />
        <StepCard
          title="路由与任务模式"
          subtitle={`${pipeline.route.query_type} | ${pipeline.route.confidence}`}
          content={routeContent}
        />
        <StepCard
          title="资源检索"
          subtitle={`${pipeline.retrieval.selected_db || "unknown"} | ${pipeline.retrieval.method}`}
          content={JSON.stringify(pipeline.retrieval, null, 2)}
        />
        <StepCard
          title="模式对齐预览"
          subtitle={`SQL DAIL 上下文 / Cypher 过滤图模式 | ${timing.schema_linking_ms}ms`}
          content={pipeline.schema_linking.linked_schema || pipeline.schema_linking.full_schema_preview}
        />
        <StepCard
          title="查询生成"
          subtitle={`${pipeline.generation.query_type} | ${timing.generation_ms}ms`}
          content={generationContent}
        />
        <StepCard
          title="查询执行"
          subtitle={`${pipeline.execution.success ? "成功" : "失败"} | 修复次数 ${pipeline.execution.retries}`}
          content={pipeline.execution.error || pipeline.execution.result_text}
        />
        <StepCard
          title="执行验证"
          subtitle={`${pipeline.verification?.verdict || "unknown"} | ${timing.verification_ms}ms`}
          content={verificationContent}
        />
        <StepCard
          title="协调器"
          subtitle={`${pipeline.coordinator?.status || "unknown"} | ${timing.coordinator_ms}ms`}
          content={coordinatorContent}
        />
      </div>
    </aside>
  );
}
