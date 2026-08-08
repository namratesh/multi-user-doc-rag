import { useCallback, useState } from "react";
import { login as loginRequest } from "../api/client";
import { ApiError } from "../api/types";

export interface Session {
  token: string;
  email: string;
  companies: string[];
}

const STORAGE_KEY = "folio.session";

function readStoredSession(): Session | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Session) : null;
  } catch {
    return null;
  }
}

export function useAuth() {
  const [session, setSession] = useState<Session | null>(readStoredSession);
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const login = useCallback(async (email: string) => {
    setIsLoggingIn(true);
    setError(null);
    try {
      const res = await loginRequest(email.trim());
      const next: Session = {
        token: res.access_token,
        email: res.email,
        companies: res.companies,
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      setSession(next);
      return next;
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.status === 401
            ? "Unrecognized clearance email."
            : err.message
          : "Could not reach the archive. Check the backend is running.";
      setError(message);
      throw err;
    } finally {
      setIsLoggingIn(false);
    }
  }, []);

  const switchReader = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setSession(null);
    setError(null);
  }, []);

  return { session, login, switchReader, isLoggingIn, error };
}
