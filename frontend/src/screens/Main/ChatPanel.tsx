import { useEffect, useRef } from "react";
import { Message } from "./Message";
import { ChatInput } from "./ChatInput";
import type { UIMessage } from "./chatTypes";
import styles from "./ChatPanel.module.css";

interface ChatPanelProps {
  messages: UIMessage[];
  isThreadLoading: boolean;
  isSending: boolean;
  onSend: (text: string) => void;
}

export function ChatPanel({ messages, isThreadLoading, isSending, onSend }: ChatPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  return (
    <section className={styles.panel}>
      <div className={styles.scroll} ref={scrollRef}>
        {isThreadLoading ? (
          <div className={styles.empty}>
            <span className="mono">Loading thread…</span>
          </div>
        ) : messages.length === 0 ? (
          <div className={styles.empty}>
            <h2 className={styles.emptyTitle}>Open inquiry</h2>
            <p className={styles.emptyBody}>
              Ask about a document you're cleared to see. Answers cite the transcript
              they're drawn from.
            </p>
          </div>
        ) : (
          <div className={styles.thread}>
            {messages.map((m) => (
              <Message key={m.id} message={m} />
            ))}
          </div>
        )}
      </div>
      <ChatInput onSend={onSend} disabled={isSending} />
    </section>
  );
}
