import type { DatabaseCatalog, DatabaseResource, Mode } from "../types/api";
import { Icon } from "./Icon";

type Props = {
  catalog?: DatabaseCatalog;
  mode: Mode;
};

function pickResources(resources: DatabaseResource[], preferred: string[]) {
  const byId = new Map(resources.map((resource) => [resource.id, resource]));
  const picked = preferred.map((id) => byId.get(id)).filter(Boolean) as DatabaseResource[];
  const seen = new Set(picked.map((resource) => resource.id));
  return [...picked, ...resources.filter((resource) => !seen.has(resource.id))].slice(0, 3);
}

function countCypherDomains(resources: DatabaseResource[]) {
  const roots = new Set(
    resources
      .map((resource) => resource.dataset_dir || resource.actual_db || resource.id)
      .filter(Boolean)
      .map((value) => String(value).split(/[\\/]/).pop() || String(value)),
  );
  return roots.size || resources.length;
}

function Metric({ label, value, wide = false }: { label: string; value: string | number; wide?: boolean }) {
  return (
    <div className={wide ? "stat wide" : "stat"}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ResourceList({ items, kind }: { items: DatabaseResource[]; kind: "sql" | "cypher" }) {
  return (
    <div className="resource-list">
      {items.map((resource) => (
        <div className="resource-row" key={resource.id}>
          <span>{resource.id}</span>
          <strong>{kind === "sql" ? `${resource.tables?.length ?? 0} tables` : resource.actual_db || "graph"}</strong>
        </div>
      ))}
    </div>
  );
}

export function AssetPanel({ catalog, mode }: Props) {
  const sqlResources = catalog?.sql ?? [];
  const cypherResources = catalog?.cypher ?? [];
  const fallbackSqlResources: DatabaseResource[] = [
    { id: "concert_singer", type: "sql", tables: ["stadium", "singer", "concert", "singer_in_concert"] },
    { id: "network_1", type: "sql", tables: ["Highschooler", "Friend", "Likes"] },
    { id: "student_transcripts_tracking", type: "sql", tables: new Array(11).fill("table") },
  ];
  const fallbackCypherResources: DatabaseResource[] = [
    { id: "concert_singer_graph", type: "cypher", actual_db: "pbconcertsinger" },
    { id: "bloom", type: "cypher", actual_db: "bloom" },
    { id: "entityres", type: "cypher", actual_db: "entityres" },
  ];
  const visibleSqlResources = sqlResources.length ? sqlResources : fallbackSqlResources;
  const visibleCypherResources = cypherResources.length ? cypherResources : fallbackCypherResources;
  const sqlResourceCount = sqlResources.length || 17;
  const cypherResourceCount = cypherResources.length || 13;
  const sqlTableCount = sqlResources.reduce((total, resource) => total + (resource.tables?.length ?? 0), 0) || 70;
  const cypherDomainCount = countCypherDomains(cypherResources) || 13;
  const sqlSamples = pickResources(visibleSqlResources, ["concert_singer", "network_1", "student_transcripts_tracking"]);
  const cypherSamples = pickResources(visibleCypherResources, ["concert_singer_graph", "bloom", "entityres", "worldcup2019"]);

  return (
    <aside className="asset-panel" aria-label="常驻资产面板">
      <header className="asset-head">
        <div className="brand-mark">
          <Icon name="layers" />
        </div>
        <div>
          <p className="eyebrow">SQL / Cypher Assets</p>
          <h1>跨源查询资产面板</h1>
        </div>
      </header>

      <div className="asset-scroll">
        <section className="asset-section">
          <div className="section-head">
            <div className="section-title">
              <Icon name="database" className="h-4 w-4" />
              SQL 资源
            </div>
            <span className="pill">SQLite</span>
          </div>
          <div className="asset-body">
            <div className="stat-grid">
              <Metric label="Resources" value={sqlResourceCount} />
              <Metric label="Tables" value={sqlTableCount} />
              <Metric label="Agent" value="SQL" wide />
            </div>
            <ResourceList items={sqlSamples} kind="sql" />
          </div>
        </section>

        <section className="asset-section">
          <div className="section-head">
            <div className="section-title">
              <Icon name="graph" className="h-4 w-4" />
              Cypher 资源
            </div>
            <span className="pill">Neo4j</span>
          </div>
          <div className="asset-body">
            <div className="stat-grid">
              <Metric label="Resources" value={cypherResourceCount} />
              <Metric label="Domains" value={`${cypherDomainCount}+`} />
              <Metric label="Agent" value="Cypher" wide />
            </div>
            <ResourceList items={cypherSamples} kind="cypher" />
          </div>
        </section>

        <section className="asset-section">
          <div className="section-head">
            <div className="section-title">
              <Icon name="shield" className="h-4 w-4" />
              安全终止状态
            </div>
            <span className="pill">{mode === "dev" ? "Developer" : "Safety"}</span>
          </div>
          <div className="asset-body">
            <div className="safe-grid">
              {[
                ["缺少必要输入", "missing_input"],
                ["跨源映射存在歧义", "ambiguous_soft_context"],
                ["无法找到可靠映射", "unresolved_soft_context"],
                ["依赖步骤被安全阻断", "blocked"],
              ].map(([label, status]) => (
                <div className="safe-row" key={status}>
                  <span>{label}</span>
                  <strong>{status}</strong>
                </div>
              ))}
            </div>
          </div>
        </section>
      </div>
    </aside>
  );
}
