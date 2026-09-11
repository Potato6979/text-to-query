import type { Mode } from "../types/api";
import { Icon } from "./Icon";

type Props = {
  mode: Mode;
  onChange: (mode: Mode) => void;
};

export function ModeSwitcher({ mode, onChange }: Props) {
  return (
    <div className="inline-flex w-[320px] shrink-0 rounded-full border border-brand-line bg-white/75 p-1 shadow-sm backdrop-blur">
      {(["dev", "user"] as const).map((item) => (
        <button
          key={item}
          className={`inline-flex min-h-10 min-w-0 flex-1 items-center justify-center gap-2 whitespace-nowrap rounded-full px-5 text-sm font-bold transition ${
            mode === item ? "bg-brand-ink text-white shadow-sm" : "text-brand-soft hover:text-brand-ink"
          }`}
          onClick={() => onChange(item)}
          type="button"
        >
          <Icon name={item === "dev" ? "route" : "database"} className="h-4 w-4" />
          {item === "dev" ? "开发者模式" : "用户模式"}
        </button>
      ))}
    </div>
  );
}
