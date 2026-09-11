# sql_pipeline.py — SQL 链路主体
# 架构参考：CHESS (arxiv 2405.16755) 和 DIN-SQL
# 核心流程：RAG 动态模式筛选 + CoT 引导生成 + 多智能体纠错闭环

import re
import json
import sqlite3
from openai import OpenAI
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    SQLITE_DB_PATH, MAX_RETRIES, MAX_RESULT_ROWS,
    LLM_TEMPERATURE, LLM_SEED, LLM_REQUEST_TIMEOUT
)
from sqlite_utils import connect_sqlite

# ─────────────────────────────────────────────────────
# 初始化 LLM 客户端
# ─────────────────────────────────────────────────────
llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def call_llm(prompt: str) -> str:
    """统一的LLM调用入口，方便后续换模型或加日志"""
    resp = llm.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
        seed=LLM_SEED,
        timeout=LLM_REQUEST_TIMEOUT,
    )
    return resp.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────
# Step 1：Schema 提取
# 统一元数据描述框架
# ─────────────────────────────────────────────────────
def extract_schema(conn) -> str:
    """
    从 SQLite 读取表结构，格式化成 LLM 容易理解的文本。
    包含：列定义、样本数据、外键关系（用于正确构造 JOIN 路径）。
    使用统一元数据描述框架。
    """
    cur = conn.cursor()

    # 获取所有表名
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cur.fetchall()]

    schema_parts = []

    # ── 第一部分：表结构 + 样本数据 ──
    for table in tables:
        # 获取列信息
        cur.execute(f"PRAGMA table_info({table})")
        columns = [(row[1], row[2]) for row in cur.fetchall()]

        # 获取3行样本数据
        cur.execute(f"SELECT * FROM {table} LIMIT 3")
        sample_rows = cur.fetchall()

        col_str = ", ".join(f"{col}({dtype})" for col, dtype in columns)
        schema_parts.append(f"表 {table}：{col_str}")

        if sample_rows:
            col_names = [c[0] for c in columns]
            for row in sample_rows:
                row_str = ", ".join(
                    f"{col}={val}" for col, val in zip(col_names, row)
                )
                schema_parts.append(f"  样本: {row_str}")

    # ── 第二部分：外键关系（明确 JOIN 路径，避免 LLM 猜错连接键）──
    fk_lines = []
    for table in tables:
        cur.execute(f"PRAGMA foreign_key_list({table})")
        fks = cur.fetchall()
        for fk in fks:
            # fk 结构: (id, seq, ref_table, from_col, to_col, ...)
            _, _, ref_table, from_col, to_col = fk[0], fk[1], fk[2], fk[3], fk[4]
            fk_lines.append(
                f"  {table}.{from_col} → {ref_table}.{to_col}"
            )

    if fk_lines:
        schema_parts.append("\n【外键关系（JOIN 时必须按此路径连接）】")
        schema_parts.extend(fk_lines)
    else:
        # 部分 SQLite 文件没有声明外键，尝试从表名和列名推断
        schema_parts.append("\n【注意】该数据库未声明外键，请根据列名语义推断 JOIN 条件")

    cur.close()
    return "\n".join(schema_parts)


# ─────────────────────────────────────────────────────
# Step 2：Schema Linking
# 基于 RAG 的动态模式筛选
# 参考：CHESS § Schema Selector
# ─────────────────────────────────────────────────────
def schema_linking(question: str, full_schema: str) -> str:
    """
    让 LLM 从完整 Schema 中筛选出和问题相关的表和列。
    避免把无关表结构塞进生成 prompt，减少干扰。
    """
    prompt = f"""你是数据库专家。根据用户问题，找出需要用到的表和列。

数据库结构（含样本数据）：
{full_schema}

用户问题：{question}

请用以下 JSON 格式返回，不要有其他内容：
{{
  "relevant_tables": ["表名1", "表名2"],
  "relevant_columns": ["表名.列名", "表名.列名"]
}}"""

    raw = call_llm(prompt)

    # 从回复中提取 JSON（防止模型多输出解释文字）
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        return full_schema  # 提取失败时降级，用完整 schema

    try:
        data = json.loads(match.group())
        tables = data.get("relevant_tables", [])
        cols   = data.get("relevant_columns", [])

        # 从完整 schema 里只保留相关行
        linked_lines = []
        for line in full_schema.split("\n"):
            if any(t in line for t in tables):
                linked_lines.append(line)

        result = "\n".join(linked_lines)
        result += f"\n\n相关列：{', '.join(cols)}"
        return result

    except json.JSONDecodeError:
        return full_schema


# ─────────────────────────────────────────────────────
# Step 3：SQL 生成（CoT 思维链）
# 面向关系模型的 CoT 引导路径生成
# 参考：DIN-SQL 任务分解 + CHESS § Candidate Generator
# ─────────────────────────────────────────────────────
def generate_sql(question: str, linked_schema: str) -> str:
    """
    用 CoT（思维链）引导 LLM 分步推理，最后生成 SQL。
    分步骤思考能有效减少多表 JOIN 时的列名错误。
    """
    prompt = f"""你是 SQLite 数据库专家。请按步骤思考后生成 SQL。

数据库相关结构：
{linked_schema}

用户问题：{question}

请按以下步骤思考：
1. 问题需要查询什么数据？
2. 涉及哪些表？需要 JOIN 吗？JOIN 条件是什么？
3. 有没有过滤条件（WHERE）？
4. 需要聚合（GROUP BY / COUNT / AVG）吗？

生成规则：
- 【最重要】如果 Schema 中有【外键关系】部分，JOIN 条件必须严格按照外键路径写，不能自行猜测连接列
- 除非问题明确要求包含没有关联数据的记录（如"包括没有...的"），否则使用 INNER JOIN 而非 LEFT JOIN
- SELECT 只返回问题明确要求的列，不要额外添加 ID 列或计数列
- 数据库是 SQLite，列名不需要双引号，直接写列名即可，例如写 name 而不是 "name"
- 不要使用 ILIKE，SQLite 不支持，字符串模糊匹配用 LIKE 代替

最后给出 SQL，用 <sql> 标签包裹：
<sql>
在这里写 SQL
</sql>"""

    raw = call_llm(prompt)

    # 提取 <sql> 标签内容
    match = re.search(r'<sql>(.*?)</sql>', raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 兜底：提取第一个 SELECT 语句
    match2 = re.search(r'(SELECT[\s\S]+?;)', raw, re.IGNORECASE)
    if match2:
        return match2.group(1).strip()

    return raw.strip()


# ─────────────────────────────────────────────────────
# Step 4：执行 SQL
# ─────────────────────────────────────────────────────
def execute_sql(conn, sql: str) -> tuple:
    """
    执行 SQL，返回 (是否成功, 结果文本或报错信息)
    """
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        col_names = [desc[0] for desc in cur.description] if cur.description else []
        cur.close()

        if not rows:
            return True, "（查询成功，结果为空）"

        # 格式化成表格文本
        header = " | ".join(col_names)
        sep    = "-" * len(header)
        lines  = [header, sep]
        for row in rows[:MAX_RESULT_ROWS]:
            lines.append(" | ".join(str(v) if v is not None else "NULL" for v in row))

        if len(rows) > MAX_RESULT_ROWS:
            lines.append(f"... 共 {len(rows)} 行，只显示前 {MAX_RESULT_ROWS} 行")

        return True, "\n".join(lines)

    except Exception as e:
        return False, str(e)


# ─────────────────────────────────────────────────────
# Step 5：纠错 Agent
# 生成、反馈和重构闭环
# 参考：CHESS § Unit Tester，DIN-SQL 自纠错
# ─────────────────────────────────────────────────────
def fix_sql(question: str, linked_schema: str, bad_sql: str, error_msg: str, attempt: int = 1) -> str:
    """
    把执行失败的 SQL 和报错信息一起交给 LLM 修正。
    这是「多智能体纠错机制」中生成Agent接收反馈的核心步骤。
    """
    prompt = f"""你生成的 SQL 执行时报错了，这是第 {attempt} 次修正尝试，请认真分析后给出和之前完全不同的修正方案。

数据库结构（注意：列名带双引号的必须在SQL中用双引号包裹，类型也已标注）：
{linked_schema}

用户问题：{question}

出错的 SQL：
{bad_sql}

SQLite 报错信息：
{error_msg}

修正要求：
1. 仔细阅读报错信息，找出根本原因
2. 【最重要】如果 Schema 中有【外键关系】部分，JOIN 条件必须严格按照外键路径，不能自行猜测
3. 数据库是 SQLite，列名不需要双引号，直接写列名即可
4. SQLite 不支持 ILIKE，字符串模糊匹配改用 LIKE
5. SQLite 不支持 :: 类型转换，改用 CAST(x AS INTEGER) 语法
6. 不要重复上次失败的 SQL，必须给出不同的修正方案

给出修正后的 SQL，用 <sql> 标签包裹：
<sql>
修正后的 SQL
</sql>"""

    raw = call_llm(prompt)
    match = re.search(r'<sql>(.*?)</sql>', raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw.strip()


# ─────────────────────────────────────────────────────
# 主流程：把所有模块串起来
# ─────────────────────────────────────────────────────
def run_sql_pipeline(question: str, db_path: str = None) -> dict:
    """
    完整 SQL 链路：
    问题 → (向量检索定位数据库) → Schema提取 → Schema Linking → SQL生成 → 执行 → (纠错重试)

    db_path 优先级：
      1. 若显式传入 db_path（评估脚本指定），直接使用
      2. 否则调用向量化 Schema 检索自动定位目标数据库
      3. 兜底使用 config.SQLITE_DB_PATH

    返回包含所有中间步骤的字典，便于调试和展示执行过程。
    """
    record = {
        "question":      question,
        "schema_linked": "",
        "sql":           "",
        "success":       False,
        "result":        "",
        "retries":       0,
        "error":         "",
        "retrieved_db":  "",
        "used_db_path":  "",
    }

    # 自动定位数据库：未指定路径时使用向量检索 Top-3 依次尝试
    # 若显式传入 db_path，封装成单元素列表，逻辑统一
    if db_path is not None:
        db_candidates = [{"db_id": "specified", "db_path": db_path}]
    else:
        try:
            from schema_retriever import retrieve_db
            retrieval     = retrieve_db(question)
            top3          = retrieval["candidates"][:3]
            db_candidates = [{"db_id": c["db_id"], "db_path": c["db_path"]} for c in top3]
            record["retrieved_db"] = retrieval["db_id"]
            print(f"[RAG] Top-3 候选库：{[c['db_id'] for c in top3]}")
        except Exception as e:
            print(f"[RAG] 检索失败，使用默认数据库：{e}")
            db_candidates = [{"db_id": "default", "db_path": SQLITE_DB_PATH}]

    print("\n" + "="*50)
    print(f"问题：{question}")
    print("─"*50)

    for db_rank, db_candidate in enumerate(db_candidates):
        current_db_path = db_candidate["db_path"]
        current_db_id   = db_candidate["db_id"]

        if db_rank > 0:
            print(f"\n[RAG] 尝试第 {db_rank+1} 候选库：{current_db_id}")

        conn = connect_sqlite(current_db_path)
        try:
            # ── Step 1：提取 Schema ──
            print(f"[Step 1] 提取 Schema（{current_db_id}）...")
            full_schema = extract_schema(conn)

            # ── Step 2：Schema Linking ──
            print("[Step 2] Schema Linking（筛选相关表列）...")
            linked = schema_linking(question, full_schema)
            record["schema_linked"] = linked
            print(f"  相关结构：\n{linked[:200]}...")

            # ── Step 3：生成 SQL ──
            print("[Step 3] 生成 SQL（CoT推理）...")
            sql = generate_sql(question, linked)
            record["sql"] = sql
            print(f"  生成SQL：{sql}")

            # ── Step 4 + 5：执行 + 纠错重试 ──
            db_success = False
            for attempt in range(MAX_RETRIES + 1):
                if attempt == 0:
                    print("[Step 4] 执行 SQL...")
                else:
                    print(f"[Step 5] 纠错重试（第 {attempt} 次）...")

                ok, output = execute_sql(conn, sql)

                if ok:
                    record["success"]      = True
                    record["result"]       = output
                    record["retries"]      = attempt
                    record["retrieved_db"] = current_db_id
                    record["used_db_path"] = current_db_path
                    db_success = True
                    print(f"  [OK] SQL executed successfully (db: {current_db_id})")
                    print(f"  Result:\n{output}")
                    break
                else:
                    record["error"] = output
                    print(f"  [ERROR] SQL execution failed: {output}")

                    # no such table → 库选错了，跳出去换下一个候选库
                    # 其他错误（语法/列名）→ 走纠错 Agent，不换库
                    if "no such table" in output.lower() and attempt == 0:
                        print("  [RAG] 检测到 no such table，尝试下一候选库")
                        break

                    if attempt < MAX_RETRIES:
                        sql = fix_sql(question, linked, sql, output, attempt + 1)
                        record["sql"] = sql
                        print(f"  纠错后SQL：{sql}")
                    else:
                        record["result"] = f"执行失败（已重试 {MAX_RETRIES} 次）：{output}"
                        print("  已达最大重试次数，放弃")

            if db_success:
                break

        finally:
            conn.close()
    return record


# ─────────────────────────────────────────────────────
# 批量测试：用 Spider 标准问题评估链路准确率
# ─────────────────────────────────────────────────────
def batch_test(questions: list) -> None:
    """
    跑一批测试问题，统计成功率。
    questions 是字符串列表，每个是一个自然语言问题。
    """
    total   = len(questions)
    success = 0

    for i, q in enumerate(questions, 1):
        print(f"\n{'='*50}")
        print(f"[{i}/{total}]")
        result = run_sql_pipeline(q)
        if result["success"]:
            success += 1

    print(f"\n{'='*50}")
    print(f"批量测试完成：{success}/{total} 成功，"
          f"成功率 {success/total*100:.1f}%")


# ─────────────────────────────────────────────────────
# 入口：直接运行这个文件做测试
# ─────────────────────────────────────────────────────
if __name__ == "__main__":

    # 单条测试——先跑这个确认链路通了
    single_question = "How many singers are there?"
    run_sql_pipeline(single_question)

    # 确认单条没问题后，取消下面的注释跑批量测试
    # test_questions = [
    #     "How many singers are there?",
    #     "What are the names of the singers from country 'France'?",
    #     "Show the name and the release year of the song by the singer whose birth year is the latest.",
    #     "What is the average age of all singers from Japan?",
    #     "How many concerts are there in year 2014?",
    #     "Show the stadium name and the number of concerts in each stadium.",
    #     "What are the names and countries of singers who performed in concerts in 2014?",
    # ]
    # batch_test(test_questions)
