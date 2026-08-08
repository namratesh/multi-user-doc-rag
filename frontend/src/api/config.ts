const API_IP = import.meta.env.VITE_API_IP ?? "127.0.0.1";
const API_PORT = import.meta.env.VITE_API_PORT ?? "8000";

export const API_BASE_URL = `http://${API_IP}:${API_PORT}`;

export function apiPath(path: string): string {
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}
