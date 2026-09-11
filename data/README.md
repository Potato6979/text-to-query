# 数据配置

此目录用于存放 SQL、Cypher 和多步跨源查询所需的数据资源。`paired_benchmark/` 包含项目内置的跨源数据，Spider 和 Mind-the-Query 数据按照以下目录结构配置。

## Spider SQL 数据

代码期望以下结构：

```text
data/spider_data/
|-- tables.json
|-- dev.json
|-- train_spider_and_others.json
|-- enc/
|   `-- train_schema-linking.jsonl
`-- database/
    |-- concert_singer/concert_singer.sqlite
    |-- car_1/car_1.sqlite
    `-- <db_id>/<db_id>.sqlite
```

SQL 索引包含 13 个数据库：`car_1`、`concert_singer`、`cre_Doc_Template_Mgt`、`dog_kennels`、`employee_hire_evaluation`、`flight_2`、`network_1`、`orchestra`、`pets_1`、`student_transcripts_tracking`、`tvshow`、`world_1` 和 `wta_1`。

从 Spider 官方发布页获取数据后保持官方目录名。DAIL 示例池默认由 `tables.json`、`train_spider_and_others.json` 与 `enc/train_schema-linking.jsonl` 构建，并把缓存写入 `schema_index/sql_dail_example_pool_cache/`。

## Mind-the-Query / Neo4j 数据

Cypher 索引包含以下逻辑数据库名：`bloom`、`entityres`、`gdsc`、`healthcare`、`legis`、`osm`、`pole`、`trolls` 和 `wwc2019`。请按所用 Neo4j 版本的官方导入方式恢复图数据，并保证数据库名与索引一致。

## Paired Benchmark

多步跨源功能使用：

```text
data/paired_benchmark/
|-- manifest.json
|-- bridge_gold_mappings.json
|-- case_schema_v1.json
|-- sql_to_cypher/
`-- cypher_to_sql/
```

该目录包含四个领域的 SQLite 子集、图记录和 Cypher 导入脚本，以及 404 条双向桥接映射。`schema_index/paired_benchmark_resources.json` 定义运行时资源目录，`case_schema_v1.json` 定义跨源查询的数据契约。

运行 paired Cypher 与跨源查询前，需要将各领域的 `.cypher` 文件导入资源目录指定的 Neo4j 命名数据库。SQLite 子集可直接通过对应的 `.sqlite` 文件访问。

## 重要说明

- 不要把下载数据、数据库密码或数据库备份直接提交到 Git。
- 如果重新生成正式 FAISS 索引，必须同时替换 `.index` 与对应的 `metadata.pkl`，不能混用不同批次。
- Pickle 文件只应从可信来源加载。
