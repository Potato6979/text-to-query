import { useEffect, useMemo, useState } from "react";

import type { DevQueryResponse, QueryResponse, UserQueryResponse } from "../types/api";
import { Icon } from "./Icon";

type RuntimeMode = "pending" | "single_sql" | "single_cypher" | "multi_step";

type Props = {
  completed?: boolean;
  response?: QueryResponse;
};

const PENDING_DELAYS = [
  { step: 1, delay: 0 },
  { step: 2, delay: 700 },
];

const SINGLE_DELAYS = [
  { step: 1, delay: 0 },
  { step: 2, delay: 700 },
  { step: 3, delay: 1400 },
];

const MULTI_DELAYS = [
  { step: 1, delay: 0 },
  { step: 2, delay: 700 },
  { step: 3, delay: 1400 },
  { step: 4, delay: 2100 },
];

function isDevResponse(response?: QueryResponse): response is DevQueryResponse {
  return Boolean(response && "pipeline" in response);
}

function isUserResponse(response?: QueryResponse): response is UserQueryResponse {
  return Boolean(response && !("pipeline" in response));
}

function asBool(value: unknown) {
  return value === true || value === "true";
}

function inferRuntimeMode(response?: QueryResponse): RuntimeMode {
  if (!response) return "pending";

  const taskMode = isDevResponse(response)
    ? response.pipeline.route.task_mode || response.pipeline.coordinator?.mode || ""
    : isUserResponse(response)
      ? response.task_mode || ""
      : "";
  const runtimeDecision = isDevResponse(response)
    ? response.pipeline.route.runtime_decision || response.runtime_decision || {}
    : isUserResponse(response)
      ? response.runtime_decision || {}
      : {};

  if (
    asBool(runtimeDecision.effective_multi_step_runtime) ||
    taskMode === "multi_step" ||
    taskMode === "multi_step_candidate" ||
    taskMode === "cross_source_candidate"
  ) {
    return "multi_step";
  }

  const queryType = isDevResponse(response) ? response.pipeline.route.query_type : response.query_type;
  return queryType === "cypher" ? "single_cypher" : "single_sql";
}

function visibleClass(currentStep: number, step: number) {
  return currentStep >= step ? "visible" : "";
}

function QueryPreview({ mode, response }: { mode: RuntimeMode; response?: QueryResponse }) {
  const query = isDevResponse(response) ? response.final_answer.query : response?.query;
  if (!query) {
    return (
      <pre className="runtime-code">{mode === "single_cypher" ? "MATCH ... RETURN ..." : "SELECT ... FROM ..."}</pre>
    );
  }
  return <pre className="runtime-code">{query}</pre>;
}

function responseSucceeded(response?: QueryResponse) {
  if (!response) return false;
  return isDevResponse(response) ? response.final_answer.success : response.success;
}

function PendingRuntime({ visibleStep }: { visibleStep: number }) {
  return (
    <>
      <article className={`runtime-card runtime-reveal ${visibleClass(visibleStep, 1)}`}>
        <div className="runtime-card-head">
          <div className="runtime-title">
            <Icon name="route" className="h-4 w-4" />
            Coordinator：任务模式分析
          </div>
          <span className="runtime-pill">Routing</span>
        </div>
        <div className="runtime-card-body">
          <div className="runtime-chip-row">
            <span>intent signals</span>
            <span>schema evidence</span>
            <span>resource candidates</span>
          </div>
          <p className="runtime-copy">
            系统正在根据自然语言意图、候选资源和 schema 证据判断该问题应进入单步查询还是多步跨源执行。
          </p>
        </div>
      </article>

      <article className={`runtime-card runtime-reveal ${visibleClass(visibleStep, 2)}`}>
        <div className="runtime-card-head">
          <div className="runtime-title">
            <Icon name="layers" className="h-4 w-4" />
            Schema Plan：候选资源准备
          </div>
          <span className="runtime-pill">Analyzing</span>
        </div>
        <div className="runtime-card-body">
          <div className="runtime-chip-row">
            <span>SQL catalog</span>
            <span>Cypher catalog</span>
            <span>grounding</span>
          </div>
          <p className="runtime-copy">等待 Coordinator 返回最终任务模式后，右侧舞台将切换为对应的单步或多步执行视图。</p>
        </div>
      </article>
    </>
  );
}

function SingleRuntime({
  visibleStep,
  mode,
  response,
}: {
  visibleStep: number;
  mode: "single_sql" | "single_cypher";
  response?: QueryResponse;
}) {
  const isCypher = mode === "single_cypher";
  return (
    <>
      <article className={`runtime-card runtime-reveal ${visibleClass(visibleStep, 1)}`}>
        <div className="runtime-card-head">
          <div className="runtime-title">
            <Icon name={isCypher ? "graph" : "database"} className="h-4 w-4" />
            STEP 1：{isCypher ? "图数据库资源定位" : "关系型资源定位"}
          </div>
          <span className="runtime-pill">{isCypher ? "Neo4j" : "SQLite"}</span>
        </div>
        <div className="runtime-card-body">
          <div className="runtime-chip-row">
            <span>task mode: single_step</span>
            <span>{isCypher ? "graph schema" : "table schema"}</span>
            <span>schema grounding</span>
          </div>
          <p className="runtime-copy">
            Coordinator 判定该问题可以在单一数据源内完成，因此不会展示跨源桥接和第二步查询面板。
          </p>
        </div>
      </article>

      <article className={`runtime-card runtime-reveal ${visibleClass(visibleStep, 2)}`}>
        <div className="runtime-card-head">
          <div className="runtime-title">
            <Icon name="code" className="h-4 w-4" />
            STEP 2：{isCypher ? "Cypher 查询生成与执行" : "SQL 查询生成与执行"}
          </div>
          <span className="runtime-pill">{isCypher ? "Cypher Agent" : "SQL Agent"}</span>
        </div>
        <div className="runtime-card-body">
          <div className="runtime-chip-row">
            <span>{isCypher ? "constraint hints" : "value hints"}</span>
            <span>generation</span>
            <span>execution</span>
          </div>
          <QueryPreview mode={mode} response={response} />
        </div>
      </article>

      <article className={`runtime-fusion runtime-reveal ${visibleClass(visibleStep, 3)}`}>
        <div className="runtime-fusion-head">
          <div className="runtime-title">
            <Icon name="shield" className="h-4 w-4" />
            STEP 3：Verification 与最终答案合成
          </div>
          <span className="runtime-pill">FAS Ready</span>
        </div>
        <table className="runtime-table">
          <thead>
            <tr>
              <th>Task Mode</th>
              <th>Agent</th>
              <th>Bridge</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>single_step</td>
              <td>{isCypher ? "Cypher Agent" : "SQL Agent"}</td>
              <td>not required</td>
              <td>{responseSucceeded(response) ? "verified" : "finished"}</td>
            </tr>
          </tbody>
        </table>
      </article>
    </>
  );
}

function MultiStepRuntime({ visibleStep }: { visibleStep: number }) {
  return (
    <>
      <article className={`runtime-card runtime-reveal ${visibleClass(visibleStep, 1)}`}>
        <div className="runtime-card-head">
          <div className="runtime-title">
            <Icon name="database" className="h-4 w-4" />
            STEP 1：关系型引擎子查询编译
          </div>
          <span className="runtime-pill">SQLite</span>
        </div>
        <div className="runtime-card-body">
          <div className="runtime-chip-row">
            <span>schema grounding</span>
            <span>value hints</span>
            <span>SQL execution</span>
          </div>
          <pre className="runtime-code">{`SELECT country_name
FROM countries
WHERE country_name = 'Sweden'`}</pre>
        </div>
      </article>

      <div className={`runtime-bridge runtime-reveal ${visibleClass(visibleStep, 2)}`}>
        <div />
        <div className="runtime-bridge-node">
          <Icon name="link" className="h-4 w-4" />
          跨源映射网关：中间实体变量解包并进入 Bridge Resolver
        </div>
        <div />
      </div>

      <article className={`runtime-card runtime-reveal ${visibleClass(visibleStep, 3)}`}>
        <div className="runtime-card-head">
          <div className="runtime-title">
            <Icon name="graph" className="h-4 w-4" />
            STEP 2：图数据库模型关联推理
          </div>
          <span className="runtime-pill">Neo4j</span>
        </div>
        <div className="runtime-card-body runtime-graph-layout">
          <div>
            <div className="runtime-chip-row">
              <span>entity mapping</span>
              <span>graph traversal</span>
              <span>topology match</span>
            </div>
            <p className="runtime-copy">将第一步得到的实体变量作为强上下文，约束图数据库侧的节点匹配和关系扩展。</p>
          </div>
          <svg className="runtime-topology" viewBox="0 0 240 118" role="img" aria-label="Graph topology preview">
            <line x1="54" y1="58" x2="122" y2="30" />
            <line x1="54" y1="58" x2="124" y2="86" />
            <line x1="122" y1="30" x2="190" y2="58" />
            <line x1="124" y1="86" x2="190" y2="58" />
            <circle cx="54" cy="58" r="17" />
            <circle cx="122" cy="30" r="14" />
            <circle cx="124" cy="86" r="14" />
            <circle cx="190" cy="58" r="18" />
            <text x="54" y="62">
              SQL
            </text>
            <text x="122" y="34">
              E
            </text>
            <text x="124" y="90">
              R
            </text>
            <text x="190" y="62">
              G
            </text>
          </svg>
        </div>
      </article>

      <article className={`runtime-fusion runtime-reveal ${visibleClass(visibleStep, 4)}`}>
        <div className="runtime-fusion-head">
          <div className="runtime-title">
            <Icon name="shield" className="h-4 w-4" />
            最终答案合成
          </div>
          <span className="runtime-pill">FAS Ready</span>
        </div>
        <table className="runtime-table">
          <thead>
            <tr>
              <th>Runtime</th>
              <th>Source Step</th>
              <th>Target Step</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>multi_step</td>
              <td>SQL variable</td>
              <td>Cypher context</td>
              <td>verified</td>
            </tr>
            <tr>
              <td>bridge</td>
              <td>intermediate entity</td>
              <td>target resource</td>
              <td>resolved</td>
            </tr>
          </tbody>
        </table>
      </article>
    </>
  );
}

export function RuntimeAnimation({ completed = false, response }: Props) {
  const runtimeMode = useMemo(() => (completed ? inferRuntimeMode(response) : "pending"), [completed, response]);
  const [visibleStep, setVisibleStep] = useState(completed ? 0 : 0);

  useEffect(() => {
    setVisibleStep(0);
    const delays = runtimeMode === "pending" ? PENDING_DELAYS : runtimeMode === "multi_step" ? MULTI_DELAYS : SINGLE_DELAYS;
    const timers = delays.map(({ step, delay }) => window.setTimeout(() => setVisibleStep(step), delay));
    return () => timers.forEach((timer) => window.clearTimeout(timer));
  }, [runtimeMode]);

  const title =
    runtimeMode === "pending"
      ? "Coordinator 正在分析任务模式"
      : runtimeMode === "multi_step"
        ? "跨源执行链路已生成"
        : "单步查询链路已生成";
  const statusLabel = runtimeMode === "pending" ? "Analyzing" : visibleStep >= (runtimeMode === "multi_step" ? 4 : 3) ? "Ready" : "Running";

  return (
    <div className="runtime-stage">
      <div className="runtime-stage-header">
        <div>
          <p className="runtime-eyebrow">{runtimeMode === "multi_step" ? "Distributed Runtime" : "Coordinator Runtime"}</p>
          <h3>{title}</h3>
        </div>
        <span className="runtime-status">{statusLabel}</span>
      </div>

      <div className="runtime-stack">
        {runtimeMode === "pending" ? <PendingRuntime visibleStep={visibleStep} /> : null}
        {runtimeMode === "single_sql" || runtimeMode === "single_cypher" ? (
          <SingleRuntime mode={runtimeMode} response={response} visibleStep={visibleStep} />
        ) : null}
        {runtimeMode === "multi_step" ? <MultiStepRuntime visibleStep={visibleStep} /> : null}
      </div>
    </div>
  );
}
