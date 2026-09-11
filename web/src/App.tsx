import { useEffect, useState } from "react";

import { AssetPanel } from "./components/AssetPanel";
import { ChatPanel } from "./components/ChatPanel";
import { InputBar } from "./components/InputBar";
import { ModeSwitcher } from "./components/ModeSwitcher";
import { PipelinePanel } from "./components/PipelinePanel";
import { API_BASE } from "./config";
import { useQuery } from "./hooks/useQuery";
import type { ChatMessage, DatabaseCatalog, DevQueryResponse, Mode, QueryResponse } from "./types/api";

const STORAGE_KEY = "text-to-query-mode";
const USER_RUNTIME_ANIMATION_MS = 2600;

function isDevResponse(response: QueryResponse): response is DevQueryResponse {
  return "pipeline" in response;
}

export default function App() {
  const [mode, setMode] = useState<Mode>("user");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [activeResponse, setActiveResponse] = useState<DevQueryResponse | undefined>();
  const [catalog, setCatalog] = useState<DatabaseCatalog | undefined>();
  const [enableMultiStepRuntime, setEnableMultiStepRuntime] = useState<boolean | undefined>();
  const [enableBridgeResolver, setEnableBridgeResolver] = useState<boolean | undefined>();
  const [isRuntimeAnimating, setIsRuntimeAnimating] = useState(false);
  const { runQuery, isLoading, error } = useQuery();
  const isDevMode = mode === "dev";
  const isBusy = isLoading || isRuntimeAnimating;

  useEffect(() => {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved === "dev" || saved === "user") {
      setMode(saved);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/api/databases`)
      .then((response) => (response.ok ? response.json() : undefined))
      .then((payload: DatabaseCatalog | undefined) => {
        if (!cancelled && payload) {
          setCatalog(payload);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setCatalog(undefined);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, mode);
    const latest = [...messages].reverse().find((message) => message.response)?.response;
    if (latest && isDevResponse(latest) && mode === "dev") {
      setActiveResponse(latest);
    } else if (mode === "user") {
      setActiveResponse(undefined);
    }
  }, [messages, mode]);

  async function handleSubmit(question: string) {
    const userMessage: ChatMessage = {
      id: `${Date.now()}-user`,
      role: "user",
      text: question,
    };
    const loadingMessage: ChatMessage = {
      id: `${Date.now()}-system`,
      role: "system",
      loading: true,
    };

    setMessages((current) => [...current, userMessage, loadingMessage]);
    const shouldAnimateRuntime = mode === "user";
    if (shouldAnimateRuntime) {
      setIsRuntimeAnimating(true);
    }
    const animationGate = shouldAnimateRuntime
      ? new Promise<void>((resolve) => {
          window.setTimeout(resolve, USER_RUNTIME_ANIMATION_MS);
        })
      : Promise.resolve();

    try {
      const devOverrides: { enableMultiStepRuntime?: boolean; enableBridgeResolver?: boolean } = {};
      if (enableMultiStepRuntime !== undefined) {
        devOverrides.enableMultiStepRuntime = enableMultiStepRuntime;
      }
      if (enableBridgeResolver !== undefined) {
        devOverrides.enableBridgeResolver = enableBridgeResolver;
      }
      const response = await runQuery(question, mode, mode === "dev" ? devOverrides : {});
      await animationGate;
      setMessages((current) =>
        current.map((message) =>
          message.id === loadingMessage.id ? { ...message, loading: false, response } : message,
        ),
      );
      if (mode === "dev" && isDevResponse(response)) {
        setActiveResponse(response);
      }
    } catch {
      await animationGate;
      setMessages((current) =>
        current.map((message) =>
          message.id === loadingMessage.id
            ? {
                ...message,
                loading: false,
                response: {
                  question,
                  success: false,
                  query_type: "unknown",
                  query: "",
                  result_text: `请求未完成。请检查浏览器 Network 中 /api/query 的状态，或确认前端正在连接 ${API_BASE}。`,
                  status: "failed",
                  status_label: "请求失败",
                  safe_terminal: false,
                  result_rows: [],
                  result_columns: [],
                },
              }
            : message,
        ),
      );
    } finally {
      if (shouldAnimateRuntime) {
        setIsRuntimeAnimating(false);
      }
    }
  }

  function handleQuickPrompt(question: string) {
    if (!isBusy) {
      void handleSubmit(question);
    }
  }

  return (
    <div className="app-shell">
      <AssetPanel catalog={catalog} mode={mode} />

      <main className="workspace">
        <header className="workspace-header">
          <div className="min-w-0">
            <p className="text-xs font-bold uppercase tracking-[0.20em] text-brand-blue">
              SQL / Cypher Multi-Agent Query Console
            </p>
            <h1 className="mt-1 text-3xl font-semibold leading-tight text-brand-ink">跨源自然语言数据库查询演示</h1>
            <p className="mt-1 max-w-3xl text-sm leading-6 text-brand-muted">
              关系数据库、图数据库、跨源桥接与最终答案合成的统一执行视图。
            </p>
          </div>
          <ModeSwitcher mode={mode} onChange={setMode} />
        </header>

        <section className="chat-console">
          <div className="chat-head">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.14em] text-brand-blue">Conversation Workspace</p>
              <h2 className="mt-1 text-lg font-semibold text-brand-ink">对话式查询工作区</h2>
            </div>
            <span className="rounded-full border border-brand-line bg-brand-paper px-3 py-1 text-xs font-bold text-brand-soft">
              {isBusy ? "Running" : messages.length ? "Result" : "Waiting"}
            </span>
          </div>

          <div className="conversation-scroll premium-scrollbar">
            <ChatPanel messages={messages} mode={mode} onPromptClick={handleQuickPrompt} />
            {isDevMode ? <PipelinePanel response={activeResponse} /> : null}
          </div>

          <div className="composer">
            <InputBar
              disabled={isBusy}
              enableBridgeResolver={enableBridgeResolver}
              enableMultiStepRuntime={enableMultiStepRuntime}
              mode={mode}
              onBridgeResolverChange={setEnableBridgeResolver}
              onMultiStepRuntimeChange={(enabled) => {
                setEnableMultiStepRuntime(enabled);
                if (enabled !== true) {
                  setEnableBridgeResolver(undefined);
                }
              }}
              onSubmit={handleSubmit}
            />
          </div>
        </section>

        {error ? (
          <div className="rounded-lg border border-[#d8bbb6] bg-[#f7eeec] px-4 py-2 text-xs text-[#8b4d42] shadow-sm">
            请求异常: {error}
          </div>
        ) : null}
      </main>
    </div>
  );
}
