type Props = {
  error?: string;
};

export function ErrorBlock({ error }: Props) {
  if (!error) return null;
  return (
    <div className="rounded-lg border border-[#d8bbb6] bg-[#f7eeec] px-4 py-3 text-sm text-[#8b4d42]">
      {error}
    </div>
  );
}
