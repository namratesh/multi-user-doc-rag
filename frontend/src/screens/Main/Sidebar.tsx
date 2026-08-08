import type { ConversationSummary } from "../../api/types";
import { Button } from "../../components/Button/Button";
import { formatRelativeTime } from "../../utils/formatRelativeTime";
import styles from "./Sidebar.module.css";

interface SidebarProps {
  conversations: ConversationSummary[];
  activeConvId: string | null;
  isLoading: boolean;
  onSelect: (convId: string) => void;
  onNewInquiry: () => void;
}

export function Sidebar({
  conversations,
  activeConvId,
  isLoading,
  onSelect,
  onNewInquiry,
}: SidebarProps) {
  return (
    <aside className={styles.sidebar}>
      <div className={styles.header}>
        <Button variant="primary" block onClick={onNewInquiry}>
          Start new inquiry
        </Button>
      </div>

      {isLoading ? (
        <div className={styles.loading}>Loading inquiries…</div>
      ) : conversations.length === 0 ? (
        <div className={styles.empty}>
          No inquiries yet. Ask about a document you're cleared to see.
        </div>
      ) : (
        <div className={styles.list}>
          {conversations.map((c) => (
            <button
              key={c.conv_id}
              type="button"
              className={`${styles.card} ${c.conv_id === activeConvId ? styles.cardActive : ""}`}
              onClick={() => onSelect(c.conv_id)}
            >
              <div className={styles.title}>{c.title || "New conversation"}</div>
              <div className={styles.timestamp}>{formatRelativeTime(c.updated_at)}</div>
            </button>
          ))}
        </div>
      )}
    </aside>
  );
}
