export interface LoginResponse {
  access_token: string;
  token_type: string;
  email: string;
  companies: string[];
}

export interface UserInfo {
  email: string;
  companies: string[];
}

export interface Citation {
  chunk_id: string;
  company_id: string;
  doc_id: string;
  fiscal_quarter: string | null;
  fiscal_year: string | null;
  speaker_name: string | null;
  score: number;
}

export type MessageRoute = "greet" | "deny" | "continue";

export interface MessageResponse {
  conv_id: string;
  answer: string;
  route: MessageRoute;
  citations: Citation[];
}

export interface ConversationSummary {
  conv_id: string;
  title: string;
  updated_at: string;
}

export interface CreateConversationResponse {
  conv_id: string;
}

export interface ThreadMessage {
  role: "user" | "assistant";
  content: string;
  citations: string[];
}

export interface ConversationThreadResponse {
  conv_id: string;
  messages: ThreadMessage[];
}

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}
