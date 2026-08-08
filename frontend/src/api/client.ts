import { apiPath } from "./config";
import {
  ApiError,
  type ConversationSummary,
  type ConversationThreadResponse,
  type CreateConversationResponse,
  type LoginResponse,
  type MessageResponse,
} from "./types";

async function request<T>(
  path: string,
  options: RequestInit & { token?: string } = {},
): Promise<T> {
  const { token, headers, ...rest } = options;

  const res = await fetch(apiPath(path), {
    ...rest,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...headers,
    },
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      // no JSON body
    }
    throw new ApiError(res.status, detail);
  }

  if (res.status === 204) {
    return undefined as T;
  }

  return res.json() as Promise<T>;
}

export function login(email: string): Promise<LoginResponse> {
  return request<LoginResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export function listConversations(
  token: string,
  signal?: AbortSignal,
): Promise<ConversationSummary[]> {
  return request<ConversationSummary[]>("/api/conversations", { token, signal });
}

export function createConversation(token: string): Promise<CreateConversationResponse> {
  return request<CreateConversationResponse>("/api/conversations", {
    method: "POST",
    token,
  });
}

export function getConversation(
  token: string,
  convId: string,
  signal?: AbortSignal,
): Promise<ConversationThreadResponse> {
  return request<ConversationThreadResponse>(`/api/conversations/${convId}`, {
    token,
    signal,
  });
}

export function sendMessage(
  token: string,
  convId: string,
  message: string,
): Promise<MessageResponse> {
  return request<MessageResponse>(`/api/conversations/${convId}/messages`, {
    method: "POST",
    token,
    body: JSON.stringify({ message }),
  });
}
