import { useState } from "react";
import { Icon } from "./Icon";

type Props = {
  title: string;
  subtitle?: string;
  content: string;
};

export function StepCard({ title, subtitle, content }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-lg border border-brand-line bg-white/85 p-4 shadow-sm">
      <button className="flex w-full items-center justify-between text-left" onClick={() => setOpen((v) => !v)} type="button">
        <div className="min-w-0">
          <p className="text-sm font-semibold text-brand-ink">{title}</p>
          {subtitle ? <p className="mt-1 truncate text-xs text-brand-muted">{subtitle}</p> : null}
        </div>
        <span className="inline-flex items-center gap-1 text-xs font-semibold text-brand-muted">
          <Icon name={open ? "check" : "terminal"} className="h-3.5 w-3.5" />
          {open ? "收起" : "展开"}
        </span>
      </button>
      {open ? <pre className="mt-3 max-h-80 overflow-y-auto whitespace-pre-wrap rounded-lg border border-brand-line bg-brand-paper p-3 text-xs leading-6 text-brand-soft">{content}</pre> : null}
    </div>
  );
}
