type Props = {
  loading: boolean;
  error?: string | null;
  scope?: string;
};

export function StatusBar({ loading, error, scope = "sql_cypher" }: Props) {
  return (
    <div className="rounded-lg border border-white/80 bg-white/75 px-4 py-2 text-xs text-brand-muted shadow-sm backdrop-blur">
      {loading
        ? "请求执行中..."
        : error
          ? `请求异常: ${error}`
          : `当前展示范围: ${scope}，已接入任务模式分析、多步执行、桥接解析、执行验证和最终答案合成。`}
    </div>
  );
}
