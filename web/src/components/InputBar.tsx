import { useState } from "react";

import type { Mode } from "../types/api";
import { Icon } from "./Icon";

type Props = {
  onSubmit: (question: string) => void;
  disabled?: boolean;
  mode: Mode;
  enableMultiStepRuntime?: boolean;
  enableBridgeResolver?: boolean;
  onMultiStepRuntimeChange: (enabled: boolean | undefined) => void;
  onBridgeResolverChange: (enabled: boolean | undefined) => void;
};

export function InputBar({
  onSubmit,
  disabled = false,
  mode,
  enableMultiStepRuntime,
  enableBridgeResolver,
  onMultiStepRuntimeChange,
  onBridgeResolverChange,
}: Props) {
  const [value, setValue] = useState("");

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-3 text-[11px] text-brand-muted">
        <span>支持 SQL、Cypher、同源多步和 SQL/Cypher 跨源桥接问题。</span>
      </div>
      {mode === "dev" ? (
        <div className="flex flex-wrap gap-4 text-xs text-brand-soft">
          <label className="inline-flex items-center gap-2">
            <input
              checked={enableMultiStepRuntime === true}
              className="h-4 w-4 rounded border-brand-line accent-brand-pine"
              disabled={disabled}
              onChange={(event) => onMultiStepRuntimeChange(event.target.checked ? true : undefined)}
              type="checkbox"
            />
            强制启用多步运行
          </label>
          <label className="inline-flex items-center gap-2">
            <input
              checked={enableBridgeResolver === true}
              className="h-4 w-4 rounded border-brand-line accent-brand-pine"
              disabled={disabled || enableMultiStepRuntime !== true}
              onChange={(event) => onBridgeResolverChange(event.target.checked ? true : undefined)}
              type="checkbox"
            />
            强制启用桥接解析器
          </label>
          <span className="text-brand-muted">未勾选时使用后端自动策略；用户模式始终不覆盖自动策略。</span>
        </div>
      ) : null}
      <form
        className="flex gap-3"
        onSubmit={(event) => {
          event.preventDefault();
          const next = value.trim();
          if (!next || disabled) return;
          onSubmit(next);
          setValue("");
        }}
      >
        <textarea
          className="min-h-[72px] flex-1 resize-none rounded-lg border border-brand-line bg-brand-paper px-5 py-3 text-sm leading-6 text-brand-ink outline-none ring-0 placeholder:text-slate-400 focus:border-brand-blue focus:shadow-[0_0_0_4px_rgba(56,90,114,0.10)]"
          disabled={disabled}
          onChange={(event) => setValue(event.target.value)}
          placeholder="输入自然语言问题，例如：In the school roster database, find the ninth-grade student named Gabriel, then use the social graph to list the student Gabriel likes."
          value={value}
        />
        <button
          className="inline-flex min-h-12 items-center justify-center gap-2 self-end rounded-lg bg-brand-ink px-5 py-3 text-sm font-bold text-white transition hover:bg-slate-950 disabled:cursor-not-allowed disabled:opacity-50"
          disabled={disabled}
          type="submit"
        >
          <Icon name="play" className="h-4 w-4" />
          发送
        </button>
      </form>
    </div>
  );
}
