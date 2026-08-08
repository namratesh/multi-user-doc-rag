import { useEffect, useRef } from "react";
import { Message } from "./Message";
import { ChatInput } from "./ChatInput";
import type { UIMessage } from "./chatTypes";
import styles from "./ChatPanel.module.css";

interface ChatPanelProps {
  messages: UIMessage[];
  companies: string[];
  isThreadLoading: boolean;
  isSending: boolean;
  onSend: (text: string) => void;
}

function formatCompanyList(companies: string[]): string {
  if (companies.length === 0) return "";
  if (companies.length === 1) return companies[0];
  if (companies.length === 2) return `${companies[0]} and ${companies[1]}`;
  return `${companies.slice(0, -1).join(", ")}, and ${companies[companies.length - 1]}`;
}

export function ChatPanel({
  messages,
  companies,
  isThreadLoading,
  isSending,
  onSend,
}: ChatPanelProps) {
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
            <h2 className={styles.emptyTitle}>Hello!</h2>
            <p className={styles.emptyBody}>
              {companies.length > 0
                ? `Ask me anything about the earnings calls for ${formatCompanyList(
                    companies,
                  )}, the companies you have access to.`
                : "You don't currently have access to any companies' earnings calls, so I won't be able to answer questions yet."}
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
