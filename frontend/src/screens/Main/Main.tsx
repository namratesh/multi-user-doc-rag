import { useCallback, useEffect, useRef, useState } from "react";
import {
  createConversation,
  getConversation,
  listConversations,
  sendMessageStream,
} from "../../api/client";
import { ApiError, type ConversationSummary } from "../../api/types";
import type { Session } from "../../auth/useAuth";
import { Header } from "./Header";
import { Sidebar } from "./Sidebar";
import { ChatPanel } from "./ChatPanel";
import type { UIMessage } from "./chatTypes";
import styles from "./Main.module.css";

interface MainProps {
  session: Session;
  onSwitchReader: () => void;
}

export function Main({ session, onSwitchReader }: MainProps) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [isLoadingConversations, setIsLoadingConversations] = useState(true);
  const [activeConvId, setActiveConvId] = useState<string | null>(null);
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [isThreadLoading, setIsThreadLoading] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { token } = session;
  const skipThreadLoadRef = useRef<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setIsLoadingConversations(true);
    listConversations(token, controller.signal)
      .then((convs) => {
        setConversations(convs);
        if (convs.length > 0) {
          setActiveConvId(convs[0].conv_id);
        }
      })
      .catch((err) => {
        if (!controller.signal.aborted) setError(describeError(err));
      })
      .finally(() => {
        if (!controller.signal.aborted) setIsLoadingConversations(false);
      });
    return () => {
      controller.abort();
    };
  }, [token]);

  useEffect(() => {
    if (!activeConvId) {
      setMessages([]);
      return;
    }
    if (skipThreadLoadRef.current === activeConvId) {
      skipThreadLoadRef.current = null;
      return;
    }
    const controller = new AbortController();
    setIsThreadLoading(true);
    getConversation(token, activeConvId, controller.signal)
      .then((thread) => {
        setMessages(
          thread.messages.map((m, i) => ({
            id: `${activeConvId}-${i}`,
            role: m.role,
            content: m.content,
            citations: m.citations,
          })),
        );
      })
      .catch((err) => {
        if (!controller.signal.aborted) setError(describeError(err));
      })
      .finally(() => {
        if (!controller.signal.aborted) setIsThreadLoading(false);
      });
    return () => {
      controller.abort();
    };
  }, [token, activeConvId]);

  const handleNewInquiry = useCallback(async () => {
    setError(null);
    try {
      const res = await createConversation(token);
      const summary: ConversationSummary = {
        conv_id: res.conv_id,
        title: "New conversation",
        updated_at: new Date().toISOString(),
      };
      setConversations((prev) => [summary, ...prev]);
      setActiveConvId(res.conv_id);
      setMessages([]);
    } catch (err) {
      setError(describeError(err));
    }
  }, [token]);

  const handleSend = useCallback(
    async (text: string) => {
      setError(null);
      let convId = activeConvId;

      try {
        if (!convId) {
          const res = await createConversation(token);
          convId = res.conv_id;
          skipThreadLoadRef.current = convId;
          setActiveConvId(convId);
          setConversations((prev) => [
            { conv_id: convId!, title: "New conversation", updated_at: new Date().toISOString() },
            ...prev,
          ]);
        }

        const userMsg: UIMessage = {
          id: crypto.randomUUID(),
          role: "user",
          content: text,
          citations: [],
        };
        const pendingMsg: UIMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "",
          citations: [],
          pending: true,
        };
        setMessages((prev) => [...prev, userMsg, pendingMsg]);
        setIsSending(true);

        await sendMessageStream(token, convId, text, {
          onDelta: (delta) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === pendingMsg.id
                  ? { ...m, pending: false, streaming: true, content: m.content + delta }
                  : m,
              ),
            );
          },
          onDone: (event) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === pendingMsg.id
                  ? {
                      id: m.id,
                      role: "assistant",
                      content: event.answer,
                      citations: event.citations,
                      restricted: event.route === "continue" && event.citations.length === 0,
                    }
                  : m,
              ),
            );
          },
        });

        setConversations((prev) => {
          const existing = prev.find((c) => c.conv_id === convId);
          const title =
            existing && existing.title !== "New conversation"
              ? existing.title
              : text.length > 60
                ? `${text.slice(0, 60)}…`
                : text;
          const updated: ConversationSummary = {
            conv_id: convId!,
            title,
            updated_at: new Date().toISOString(),
          };
          return [updated, ...prev.filter((c) => c.conv_id !== convId)];
        });
      } catch (err) {
        setError(describeError(err));
        setMessages((prev) => prev.filter((m) => !m.pending && !m.streaming));
      } finally {
        setIsSending(false);
      }
    },
    [activeConvId, token],
  );

  return (
    <div className={styles.app}>
      <Header email={session.email} companies={session.companies} onSwitchReader={onSwitchReader} />
      {error && <div className={styles.errorBanner}>{error}</div>}
      <div className={styles.body}>
        <Sidebar
          conversations={conversations}
          activeConvId={activeConvId}
          isLoading={isLoadingConversations}
          onSelect={setActiveConvId}
          onNewInquiry={handleNewInquiry}
        />
        <ChatPanel
          messages={messages}
          companies={session.companies}
          isThreadLoading={isThreadLoading}
          isSending={isSending}
          onSend={handleSend}
        />
      </div>
    </div>
  );
}

function describeError(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  return "Could not reach the archive. Check the backend is running.";
}
