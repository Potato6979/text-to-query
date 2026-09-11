import { Icon } from "./Icon";

type Props = {
  code: string;
  language: "sql" | "cypher";
};

export function QueryBlock({ code, language }: Props) {
  if (!code) return null;

  return (
    <div className="overflow-hidden rounded-lg border border-slate-800 bg-[#111a20]">
      <div className="flex items-center gap-2 border-b border-slate-800 px-4 py-2 text-[11px] font-bold uppercase tracking-[0.18em] text-slate-400">
        <Icon name="code" className="h-4 w-4" />
        {language}
      </div>
      <pre className="overflow-x-auto p-4 text-sm leading-6 text-slate-100">
        <code>{code}</code>
      </pre>
    </div>
  );
}
