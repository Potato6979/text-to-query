import type { ChatMessage, Mode } from "../types/api";
import { Icon } from "./Icon";
import { RuntimeAnimation } from "./RuntimeAnimation";
import { SystemMessage } from "./SystemMessage";

type Props = {
  messages: ChatMessage[];
  mode: Mode;
  onPromptClick?: (question: string) => void;
};

const QUICK_PROMPTS = [
  "Find the friends of students whose age is above the average.",
  "Which players scored goals for Sweden?",
  "In the school roster database, find the ninth-grade student named Gabriel, then use the social graph to list the student Gabriel likes.",
];

function EmptyStage({ onPromptClick }: { onPromptClick?: (question: string) => void }) {
  return (
    <div className="flex min-h-full items-center justify-center px-4 py-12">
      <div className="w-full max-w-2xl rounded-lg border border-brand-line bg-white/85 p-8 shadow-sm">
        <div className="mx-auto grid h-11 w-11 place-items-center rounded-full border border-brand-line bg-brand-paper text-brand-blue">
          <Icon name="route" className="h-5 w-5" />
        </div>
        <h2 className="mt-5 text-center text-xl font-semibold text-brand-ink">等待自然语言问题</h2>
        <p className="mx-auto mt-3 max-w-xl text-center text-sm leading-6 text-brand-muted">
          在下方输入问题后，系统会在当前对话区展示路由结果、生成查询、结构化结果与最终答案。
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-3">
          {QUICK_PROMPTS.map((prompt) => (
            <button
              className="rounded-full border border-brand-line bg-brand-paper px-4 py-2 text-sm leading-5 text-brand-soft transition hover:border-brand-blue hover:text-brand-ink"
              key={prompt}
              onClick={() => onPromptClick?.(prompt)}
              type="button"
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export function ChatPanel({ messages, mode, onPromptClick }: Props) {
  if (!messages.length) {
    return <EmptyStage onPromptClick={onPromptClick} />;
  }

  return (
    <div className="space-y-4 px-1 py-1">
      {messages.map((message) =>
        message.role === "user" ? (
          <div key={message.id} className="flex justify-end">
            <div className="max-w-[76%] rounded-lg border border-brand-line bg-white px-5 py-4 text-sm leading-6 text-brand-ink shadow-sm">
              {message.text}
            </div>
          </div>
        ) : (
          <div key={message.id} className="flex justify-start">
            <div className="w-full max-w-[92%]">
              {mode === "user" && (message.loading || message.response) ? (
                <RuntimeAnimation completed={!message.loading && Boolean(message.response)} response={message.response} />
              ) : null}
              <SystemMessage loading={message.loading} mode={mode} response={message.response} />
            </div>
          </div>
        ),
      )}
    </div>
  );
}
