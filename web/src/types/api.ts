export type Mode = "dev" | "user";

export type QueryRequest = {
  question: string;
  mode: Mode;
  enable_multi_step_runtime?: boolean;
  enable_bridge_resolver?: boolean;
};

export type QueryResponse = DevQueryResponse | UserQueryResponse;

export type SystemInfo = {
  scope: string;
  components?: string[];
  safe_terminal_statuses?: string[];
};

export type DatabaseResource = {
  id: string;
  type: "sql" | "cypher";
  path?: string;
  dataset_dir?: string;
  actual_db?: string;
  tables?: string[];
  labels?: string[];
};

export type DatabaseCatalog = {
  scope: string;
  note?: string;
  sql: DatabaseResource[];
  cypher: DatabaseResource[];
};

export type UserQueryResponse = {
  question: string;
  system?: SystemInfo;
  success: boolean;
  status?: string;
  status_label?: string;
  safe_terminal?: boolean;
  failure_type?: string;
  query_type: string;
  task_mode?: string;
  runtime_decision?: Record<string, unknown>;
  result_text: string;
  query: string;
  result_rows?: Array<Array<string | number | boolean | null | object>>;
  result_columns?: string[];
};

export type DevQueryResponse = {
  question: string;
  system?: SystemInfo;
  pipeline: {
    route: {
      query_type: string;
      confidence: string;
      reason: string;
      task_mode?: string;
      uncertainty_source?: string;
      should_keep_backup_route?: boolean;
      schema_plan_context?: Record<string, unknown>;
      schema_plan_proposals?: Array<Record<string, unknown>>;
      multi_step_plan?: Record<string, unknown>;
      multi_step_runtime?: Record<string, unknown>;
      runtime_decision?: Record<string, unknown>;
      planning_status?: string;
      execution_policy?: Record<string, unknown>;
      clarification?: Record<string, unknown>;
      low_confidence?: boolean;
      candidates?: Array<{
        query_type: string;
        score?: number;
        confidence?: string;
        reason?: string;
        candidate_id?: string;
        rank?: number;
        evidence_summary?: Record<string, unknown>;
      }>;
      query_type_candidates?: string[];
      route_plan?: {
        plan_version: number;
        task_mode: string;
        confidence?: string;
        task_analysis?: Record<string, unknown>;
        planning_status?: string;
        execution_policy?: Record<string, unknown>;
        clarification?: Record<string, unknown>;
        selected_candidate_id: string;
        selected_query_type: string;
        should_keep_backup_route: boolean;
        uncertainty_source?: string;
        schema_plan_context?: Record<string, unknown>;
        schema_plan_proposals?: Array<Record<string, unknown>>;
        proposal_execution_policy?: Record<string, unknown>;
        multi_step_plan?: Record<string, unknown>;
        multi_step_runtime_policy?: Record<string, unknown>;
        candidates: Array<{
          candidate_id: string;
          rank: number;
          query_type: string;
          score?: number;
          confidence?: string;
          reason?: string;
          evidence_summary?: Record<string, unknown>;
          plan_proposal?: Record<string, unknown>;
          candidate_source?: string;
          status?: string;
          last_detail?: string;
          last_loop_count?: number;
          last_verdict?: string;
          last_decision?: string;
          last_error?: string;
          last_generation_feedback?: Record<string, unknown>;
          attempt_history?: Array<Record<string, unknown>>;
          retrieval?: {
            method?: string;
            selected_db?: string;
            selected_path?: string;
            score?: number | null;
            top_candidates?: Array<Record<string, unknown>>;
          };
        }>;
        evidence?: Record<string, unknown>;
      };
      signals?: {
        intent_signals?: Record<string, unknown>;
        entity_hints?: Record<string, unknown>;
        modality_scores?: Record<string, unknown>;
        retrieval_hints?: Record<string, unknown>;
      };
    };
    retrieval: {
      method: string;
      top3_candidates: string[];
      candidates: Array<Record<string, unknown>>;
      selected_db: string;
      score?: number | null;
    };
    schema_linking: {
      full_schema_preview: string;
      full_schema: string;
      linked_schema: string;
    };
    generation: {
      query: string;
      query_type: string;
      cot_reasoning: string;
      task_analysis?: Record<string, unknown>;
      nested_logic_guidance?: string;
      generation_contract?: Record<string, unknown>;
      generation_feedback?: Record<string, unknown>;
      route_proposal?: Record<string, unknown>;
      proposal_target_resource?: Record<string, unknown>;
      llm_trace: Array<{
        label: string;
        prompt: string;
        response: string;
      }>;
    };
    execution: {
      success: boolean;
      result_text: string;
      result_rows: Array<Array<string | number | boolean | null | object>>;
      result_columns: string[];
      error: string;
      retries: number;
      retry_history: Array<Record<string, unknown>>;
      row_count: number;
    };
    verification?: {
      verdict: string;
      failure_type?: string;
      confidence?: string;
      reason?: string;
      suggested_action?: string;
      checks?: Record<string, unknown>;
      contract?: Record<string, unknown>;
      generation_contract_check?: Record<string, unknown>;
    };
    coordinator?: {
      task_id: string;
      mode: string;
      status: string;
      active_step_id: string;
      selected_candidate_id?: string;
      multi_step_plan_status?: string;
      multi_step_runtime_status?: string;
      loop_count: number;
      max_loops: number;
      step_retry_count: number;
      reroute_count: number;
      decision_history: Array<Record<string, unknown>>;
    };
  };
  runtime_decision?: Record<string, unknown>;
  final_answer: {
    success: boolean;
    status?: string;
    status_label?: string;
    safe_terminal?: boolean;
    query: string;
    result_text: string;
  };
  timing: Record<string, number>;
};

export type ChatMessage = {
  id: string;
  role: "user" | "system";
  question?: string;
  text?: string;
  loading?: boolean;
  response?: QueryResponse;
};
