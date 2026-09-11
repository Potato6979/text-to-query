<a id="chinese"></a>

# Text-to-Query 多智能体查询系统

简体中文 | [English](#english)

Text-to-Query 是一个面向 SQLite 与 Neo4j 的自然语言查询系统。系统接收用户问题，自动判断查询类型，检索相关 Schema，生成并执行 SQL 或 Cypher，在执行失败时尝试修复查询，并对结果进行验证和自然语言整理。

项目支持 SQL、Cypher 以及跨数据源多步查询。FastAPI 后端负责查询编排和数据访问，React 前端提供用户视图与开发视图。`main.py` 中的 MQL 和向量查询函数为兼容接口，当前未实现对应查询后端。

## 主要功能

- 根据问题语义自动选择 SQL、Cypher 或多步跨源查询
- 使用 FAISS 与 Sentence Transformers 检索相关数据库 Schema
- 使用 DAIL 风格的示例选择增强 SQL 生成
- 生成、执行和修复 SQL/Cypher 查询
- 检查查询结果形态，并通过 Verification 模块验证结果
- 规划具有前后依赖关系的多步查询
- 使用 Bridge Resolver 在 SQLite 与 Neo4j 资源之间解析实体映射
- 通过 FastAPI 提供查询、健康检查和资源目录接口
- 通过 React + Vite 页面展示答案、查询语句、执行步骤和运行状态

## 系统流程

```text
Natural-language question
        |
        v
Task analysis and schema-aware routing
        |
        +--> SQL pipeline ------> SQLite
        |
        +--> Cypher pipeline ---> Neo4j
        |
        `--> Multi-step planner -> Bridge Resolver -> SQL/Cypher steps
                                      |
                                      v
                         Verification and answer synthesis
```

## 技术栈

- Python 3.11、FastAPI、Pydantic
- OpenAI-compatible API 客户端，默认使用 DeepSeek 配置
- SQLite、Neo4j
- FAISS、NumPy、Sentence Transformers
- React 18、TypeScript、Vite、Tailwind CSS

推荐环境为 Python 3.11、Node.js 22 和 npm 10。Python 依赖版本见 `requirements.txt`，前端依赖版本见 `web/package-lock.json`。

## 目录结构

```text
.
|-- api_server.py                 # FastAPI 服务入口
|-- coordinator.py                # 多智能体协调与运行决策
|-- router.py / routing_*.py      # 查询路由、评分和保护逻辑
|-- sql_*.py                      # SQL 检索、生成、修复与执行
|-- cypher_*.py                   # Cypher 检索、生成、修复与执行
|-- multi_step_*.py               # 多步查询规划与运行时
|-- bridge_resolver.py            # 跨数据源实体桥接
|-- verification.py              # 查询结果验证
|-- schema_index/                 # 预构建 Schema 检索索引
|-- data/                         # 数据配置说明与 paired 数据资源
|-- tools/check_project.py        # 项目结构与依赖自检
|-- web/                          # React + Vite 前端
|-- requirements.txt              # Python 依赖
`-- .env.example                  # 后端环境变量示例
```

## 安装

后端（PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

前端：

```powershell
cd web
npm ci
```

`tiktoken` 为可选依赖；未安装时系统使用近似 token 计数。首次执行向量检索时，Sentence Transformers 可能下载 `paraphrase-multilingual-MiniLM-L12-v2`；使用 embedding 类型的 DAIL 示例选择时还可能下载 `all-mpnet-base-v2`。

## 配置

设置 LLM API Key。Cypher 查询还需要可访问的 Neo4j 实例：

```powershell
$env:DEEPSEEK_API_KEY = "your-api-key"
$env:DEEPSEEK_MODEL = "your-model-name"
$env:NEO4J_URI = "neo4j://127.0.0.1:7687"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASSWORD = "your-password"
```

`.env.example` 列出了后端变量，`web/.env.example` 列出了前端变量。当前 Python 代码从系统环境读取配置，不会自动加载 `.env`。不要提交真实 API Key、Token、密码或 `.env` 文件。

## 数据配置

完整的数据目录说明见 [data/README.md](data/README.md)。

- SQL Schema 索引对应 Spider 1.0 数据。将 `tables.json`、问题文件和 `database/<db_id>/<db_id>.sqlite` 放入 `data/spider_data/`。数据可从 [Spider 官方项目](https://github.com/taoyds/spider) 或 [Yale Spider 页面](https://yale-lily.github.io/spider) 获取。
- Cypher Schema 索引对应 Mind-the-Query 图数据。按照 [Mind-the-Query 官方项目](https://github.com/endeavorXx/Mind-the-Query) 的数据说明，将目标图导入本地 Neo4j，并保持数据库名称与索引一致。
- `data/paired_benchmark/` 提供 `concert_singer`、`network_1`、`wta_1` 和 `star_wars` 四个领域的 SQLite 子集、图记录、Cypher 导入脚本、桥接映射和数据契约，用于跨源查询。

预构建索引必须与实际数据库配套。API 可以在外部数据库未全部就绪时启动并返回资源目录，但执行查询要求被选中的 SQLite 文件或 Neo4j 数据库可用。

## 运行

在仓库根目录启动后端，因为 Schema 索引使用相对路径加载：

```powershell
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

另开终端启动前端：

```powershell
cd web
npm run dev
```

浏览器访问 `http://127.0.0.1:3000`。连接其他后端地址时，在 `web/.env.local` 中设置：

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## API 示例

检查服务状态和资源目录：

```powershell
curl.exe http://127.0.0.1:8000/api/health
curl.exe http://127.0.0.1:8000/api/databases
```

执行自然语言查询：

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/query `
  -H "Content-Type: application/json" `
  -d '{"question":"How many singers are there?","mode":"user"}'
```

`mode` 可设置为 `user` 或 `developer`。开发模式会返回更完整的路由、检索、生成、执行和验证信息。

## 项目自检

```powershell
python tools/check_project.py
```

该命令检查 Python 语法与本地模块引用、必要依赖、Schema 索引、paired 数据资源、前端入口文件和许可证文件。缺少外部数据库或运行凭据时会输出对应提示。

## 部署注意事项

- Cypher 查询要求 Neo4j 中存在与 Schema 索引名称一致的数据库。
- FastAPI 默认允许任意 CORS 来源，部署时应限制为实际前端域名。
- 资源目录接口可能返回数据库路径，部署时应根据安全要求过滤服务器路径。
- Pickle 索引文件只应从可信来源加载。

## 许可证

项目自有代码和内容采用 [MIT License](LICENSE)。第三方数据、衍生数据和参考实现受各自上游许可证约束，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

---

<a id="english"></a>

# Text-to-Query Multi-Agent Query System

[简体中文](#chinese) | English

Text-to-Query is a natural-language query system for SQLite and Neo4j. It classifies each request, retrieves relevant schemas, generates and executes SQL or Cypher, repairs failed queries, verifies the result, and produces a natural-language answer.

The project supports SQL, Cypher, and multi-step queries across data sources. A FastAPI backend coordinates query execution, while a React frontend provides user and developer views. The MQL and vector-query functions in `main.py` are compatibility interfaces; their corresponding query backends are not implemented.

## Features

- Automatic selection of SQL, Cypher, or multi-step cross-source execution
- Schema retrieval with FAISS and Sentence Transformers
- DAIL-style example selection for SQL generation
- SQL/Cypher generation, execution, and repair
- Result-shape checks and verification
- Planning for multi-step queries with dependent steps
- Entity mapping between SQLite and Neo4j resources through Bridge Resolver
- FastAPI endpoints for queries, health checks, and the resource catalog
- React + Vite views for answers, generated queries, execution steps, and runtime state

## Architecture

```text
Natural-language question
        |
        v
Task analysis and schema-aware routing
        |
        +--> SQL pipeline ------> SQLite
        |
        +--> Cypher pipeline ---> Neo4j
        |
        `--> Multi-step planner -> Bridge Resolver -> SQL/Cypher steps
                                      |
                                      v
                         Verification and answer synthesis
```

## Technology Stack

- Python 3.11, FastAPI, and Pydantic
- OpenAI-compatible API client, configured for DeepSeek by default
- SQLite and Neo4j
- FAISS, NumPy, and Sentence Transformers
- React 18, TypeScript, Vite, and Tailwind CSS

The recommended environment is Python 3.11, Node.js 22, and npm 10. Python versions are pinned in `requirements.txt`; frontend versions are pinned in `web/package-lock.json`.

## Repository Layout

```text
.
|-- api_server.py                 # FastAPI service entry point
|-- coordinator.py                # Multi-agent coordination and runtime decisions
|-- router.py / routing_*.py      # Query routing, scoring, and guards
|-- sql_*.py                      # SQL retrieval, generation, repair, and execution
|-- cypher_*.py                   # Cypher retrieval, generation, repair, and execution
|-- multi_step_*.py               # Multi-step planning and runtime
|-- bridge_resolver.py            # Cross-source entity resolution
|-- verification.py              # Result verification
|-- schema_index/                 # Prebuilt schema retrieval indexes
|-- data/                         # Data setup guide and paired resources
|-- tools/check_project.py        # Project structure and dependency check
|-- web/                          # React + Vite frontend
|-- requirements.txt              # Python dependencies
`-- .env.example                  # Backend environment variable template
```

## Installation

Backend on PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Frontend:

```powershell
cd web
npm ci
```

`tiktoken` is optional; without it, the system uses approximate token counting. Sentence Transformers may download `paraphrase-multilingual-MiniLM-L12-v2` on first use. Embedding-based DAIL example selection may also download `all-mpnet-base-v2`.

## Configuration

Set an LLM API key. Cypher queries also require an accessible Neo4j instance:

```powershell
$env:DEEPSEEK_API_KEY = "your-api-key"
$env:DEEPSEEK_MODEL = "your-model-name"
$env:NEO4J_URI = "neo4j://127.0.0.1:7687"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASSWORD = "your-password"
```

`.env.example` lists backend variables, and `web/.env.example` lists frontend variables. The Python application reads the system environment and does not automatically load `.env`. Never commit real API keys, tokens, passwords, or `.env` files.

## Data Setup

See [data/README.md](data/README.md) for the complete directory layout.

- The SQL schema indexes correspond to Spider 1.0. Place `tables.json`, question files, and `database/<db_id>/<db_id>.sqlite` under `data/spider_data/`. Obtain the data from the [official Spider repository](https://github.com/taoyds/spider) or the [Yale Spider page](https://yale-lily.github.io/spider).
- The Cypher schema indexes correspond to Mind-the-Query graph data. Follow the [Mind-the-Query repository](https://github.com/endeavorXx/Mind-the-Query) instructions to import the target graphs into Neo4j, preserving the database names used by the indexes.
- `data/paired_benchmark/` provides SQLite subsets, graph records, Cypher import scripts, bridge mappings, and data contracts for the `concert_singer`, `network_1`, `wta_1`, and `star_wars` domains.

Prebuilt indexes must match the actual databases. The API can start and return its resource catalog before every external database is available, but query execution requires the selected SQLite file or Neo4j database.

## Running

Start the backend from the repository root because schema indexes are loaded through relative paths:

```powershell
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal:

```powershell
cd web
npm run dev
```

Open `http://127.0.0.1:3000`. To use another backend address, set the following in `web/.env.local`:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## API Example

Check service health and the resource catalog:

```powershell
curl.exe http://127.0.0.1:8000/api/health
curl.exe http://127.0.0.1:8000/api/databases
```

Run a natural-language query:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/query `
  -H "Content-Type: application/json" `
  -d '{"question":"How many singers are there?","mode":"user"}'
```

`mode` accepts `user` or `developer`. Developer mode returns more detailed routing, retrieval, generation, execution, and verification information.

## Project Check

```powershell
python tools/check_project.py
```

This command checks Python syntax and local module references, required dependencies, schema indexes, paired data resources, frontend entry files, and license files. It reports unavailable external databases or runtime credentials.

## Deployment Notes

- Cypher execution requires Neo4j databases whose names match the schema indexes.
- FastAPI allows all CORS origins by default; restrict the configuration to the deployed frontend origins.
- The resource catalog may return database paths; filter server paths according to deployment security requirements.
- Load pickle index files only from trusted sources.

## License

Original project code and content are licensed under the [MIT License](LICENSE). Third-party data, derivative data, and referenced implementations remain subject to their upstream licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
