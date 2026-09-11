import type { DevQueryResponse, QueryResponse } from "../types/api";
import { ErrorBlock } from "./ErrorBlock";
import { Icon } from "./Icon";
import { QueryBlock } from "./QueryBlock";
import { ResultTable } from "./ResultTable";

type Props = {
  response?: QueryResponse;
  loading?: boolean;
  mode: "dev" | "user";
};

function isDevResponse(response: QueryResponse): response is DevQueryResponse {
  return "pipeline" in response;
}

function labelQueryType(queryType: string) {
  const labels: Record<string, string> = {
    sql: "SQL 查询",
    cypher: "Cypher 查询",
    mql: "MQL 占位",
    vector: "向量检索占位",
    unknown: "未知类型",
  };
  return labels[queryType] || queryType;
}

export function SystemMessage({ response, loading = false, mode }: Props) {
  if (loading) {
    if (mode === "user") return null;
    return <div className="rounded-lg border border-brand-line bg-white px-5 py-4 text-sm text-brand-muted shadow-sm">系统正在处理...</div>;
  }

  if (!response) return null;

  const query = isDevResponse(response) ? response.final_answer.query : response.query;
  const queryType = isDevResponse(response) ? response.pipeline.route.query_type : response.query_type;
  const rows = isDevResponse(response) ? response.pipeline.execution.result_rows : response.result_rows;
  const columns = isDevResponse(response) ? response.pipeline.execution.result_columns : response.result_columns;
  const error = isDevResponse(response) ? response.pipeline.execution.error : "";
  const resultText = isDevResponse(response) ? response.final_answer.result_text : response.result_text;
  const status = isDevResponse(response) ? response.final_answer.status : response.status;
  const statusLabel = isDevResponse(response) ? response.final_answer.status_label : response.status_label;
  const success = isDevResponse(response) ? response.final_answer.success : response.success;
  const taskMode = isDevResponse(response) ? response.pipeline.route.task_mode : undefined;
  const selectedDb = isDevResponse(response) ? response.pipeline.retrieval.selected_db : undefined;
  const hasNarrativeAnswer = /^(Answer for:|Status:|Bridge \/ mapping limitations)/m.test(resultText || "");
  const isSafeTerminal = ["blocked", "missing_input", "ambiguous_soft_context", "unresolved_soft_context"].includes(
    String(status || "").toLowerCase(),
  );

  return (
    <div className="space-y-4 rounded-lg border border-brand-line bg-white/85 px-5 py-5 shadow-sm">
      <div className="flex flex-wrap gap-2 text-xs text-brand-muted">
        <span className="inline-flex items-center gap-1 rounded-full border border-brand-line bg-brand-paper px-3 py-1">
          <Icon name="route" className="h-3.5 w-3.5" />
          路由: {labelQueryType(queryType)}
        </span>
        {taskMode ? <span className="rounded-full border border-brand-line bg-brand-paper px-3 py-1">任务模式: {taskMode}</span> : null}
        {status ? (
          <span className={`rounded-full border px-3 py-1 ${isSafeTerminal ? "border-[#d8c490] bg-[#f4eddc] text-[#765f2f]" : "border-[#cadbd0] bg-[#e8f0eb] text-brand-pine"}`}>
            {statusLabel || status}
          </span>
        ) : null}
        <span className={`rounded-full border px-3 py-1 ${success ? "border-[#cadbd0] bg-[#e8f0eb] text-brand-pine" : "border-brand-line bg-brand-paper text-brand-soft"}`}>
          {success ? "已生成最终答案" : isSafeTerminal ? "安全终止" : "未完成"}
        </span>
        {mode === "dev" && selectedDb ? <span className="rounded-full border border-brand-line bg-brand-paper px-3 py-1">数据库: {selectedDb}</span> : null}
      </div>
      <QueryBlock code={query} language={queryType === "cypher" ? "cypher" : "sql"} />
      {hasNarrativeAnswer || isSafeTerminal ? (
        <pre className="whitespace-pre-wrap rounded-lg border border-brand-line bg-brand-paper p-4 text-sm leading-6 text-brand-soft">{resultText}</pre>
      ) : null}
      <ResultTable columns={columns} rows={rows} />
      {!rows?.length && !hasNarrativeAnswer && !isSafeTerminal ? (
        <pre className="whitespace-pre-wrap text-sm leading-6 text-brand-soft">{resultText}</pre>
      ) : null}
      <ErrorBlock error={error} />
    </div>
  );
}
