import { useState } from "react";

import { API_BASE } from "../config";
import type { Mode, QueryResponse } from "../types/api";

// 中文函数说明：
// - useQuery：封装前端调用 /api/query 的状态管理。
// - runQuery：发送自然语言问题、返回模式和可选 dev override。
// - enableMultiStepRuntime / enableBridgeResolver 只有显式提供时才发送；用户模式不发送 false，从而不覆盖后端 auto policy。

export function useQuery() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runQuery(
    question: string,
    mode: Mode,
    options: { enableMultiStepRuntime?: boolean; enableBridgeResolver?: boolean } = {},
  ): Promise<QueryResponse> {
    setIsLoading(true);
    setError(null);

    try {
      const body: {
        question: string;
        mode: Mode;
        enable_multi_step_runtime?: boolean;
        enable_bridge_resolver?: boolean;
      } = {
        question,
        mode,
      };
      if (options.enableMultiStepRuntime !== undefined) {
        body.enable_multi_step_runtime = options.enableMultiStepRuntime;
      }
      if (options.enableBridgeResolver !== undefined) {
        body.enable_bridge_resolver = options.enableBridgeResolver;
      }

      const response = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }

      return (await response.json()) as QueryResponse;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown request error";
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }

  return { runQuery, isLoading, error };
}
