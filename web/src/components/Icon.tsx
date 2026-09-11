type IconName =
  | "check"
  | "code"
  | "database"
  | "graph"
  | "layers"
  | "link"
  | "play"
  | "route"
  | "shield"
  | "terminal";

type Props = {
  name: IconName;
  className?: string;
};

const PATHS: Record<IconName, JSX.Element> = {
  check: <path d="M20 6 9 17l-5-5" />,
  code: (
    <>
      <path d="m8 9-4 3 4 3" />
      <path d="m16 9 4 3-4 3" />
      <path d="m14 5-4 14" />
    </>
  ),
  database: (
    <>
      <ellipse cx="12" cy="5" rx="7" ry="3" />
      <path d="M5 5v14c0 1.7 3.1 3 7 3s7-1.3 7-3V5" />
      <path d="M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3" />
    </>
  ),
  graph: (
    <>
      <circle cx="6" cy="7" r="3" />
      <circle cx="18" cy="6" r="3" />
      <circle cx="16" cy="18" r="3" />
      <path d="M9 7h6M8 9l6 7M18 9l-1 6" />
    </>
  ),
  layers: (
    <>
      <path d="m12 3 9 5-9 5-9-5 9-5Z" />
      <path d="m21 12-9 5-9-5" />
      <path d="m21 16-9 5-9-5" />
    </>
  ),
  link: (
    <>
      <path d="M10 13a5 5 0 0 0 7.1 0l2-2A5 5 0 0 0 12 3.9l-1.2 1.2" />
      <path d="M14 11a5 5 0 0 0-7.1 0l-2 2A5 5 0 0 0 12 20.1l1.2-1.2" />
    </>
  ),
  play: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="m10 8 6 4-6 4V8Z" />
    </>
  ),
  route: (
    <>
      <circle cx="6" cy="18" r="3" />
      <circle cx="18" cy="6" r="3" />
      <path d="M9 18h3a4 4 0 0 0 4-4V9" />
    </>
  ),
  shield: (
    <>
      <path d="m12 3 7 3v5c0 5-3 8.5-7 10-4-1.5-7-5-7-10V6l7-3Z" />
      <path d="m9 12 2 2 4-5" />
    </>
  ),
  terminal: (
    <>
      <rect x="3" y="5" width="18" height="14" rx="2" />
      <path d="m7 9 3 3-3 3M12 15h5" />
    </>
  ),
};

export function Icon({ name, className = "h-[18px] w-[18px]" }: Props) {
  return (
    <svg
      aria-hidden="true"
      className={`shrink-0 text-[#7d7d85] ${className}`}
      fill="none"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.5"
      viewBox="0 0 24 24"
    >
      {PATHS[name]}
    </svg>
  );
}
