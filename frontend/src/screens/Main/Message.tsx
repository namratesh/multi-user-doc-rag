import { Stamp } from "../../components/Stamp/Stamp";
import { citationLabel, type UIMessage } from "./chatTypes";
import styles from "./Message.module.css";

export function Message({ message }: { message: UIMessage }) {
  if (message.role === "user") {
    return (
      <div className={`${styles.row} ${styles.user}`}>
        <div className={styles.userBubble}>{message.content}</div>
      </div>
    );
  }

  if (message.pending) {
    return (
      <div className={`${styles.row} ${styles.assistant}`}>
        <span className={styles.pendingLabel}>Retrieving…</span>
      </div>
    );
  }

  return (
    <div className={`${styles.row} ${styles.assistant}`}>
      <div className={styles.assistantBlock}>
        {message.restricted && (
          <div className={styles.restrictedRow}>
            <Stamp tone="restricted" size="inline" lines={["Restricted"]} animateKey={message.id} />
          </div>
        )}
        <p className={message.restricted ? styles.restrictedText : styles.assistantText}>
          {message.restricted
            ? "Nothing in your cleared documents covers that."
            : message.content}
        </p>
        {message.citations.length > 0 && (
          <div className={styles.citations}>
            {message.citations.map((c, i) => (
              <button
                key={typeof c === "string" ? c : c.chunk_id}
                type="button"
                className={styles.citationTab}
              >
                {citationLabel(c) || `Source ${i + 1}`}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
