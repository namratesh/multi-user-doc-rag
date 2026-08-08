import { useState } from "react";
import { Stamp } from "../../components/Stamp/Stamp";
import { citationLabel, isFullCitation, type UIMessage } from "./chatTypes";
import styles from "./Message.module.css";

export function Message({ message }: { message: UIMessage }) {
  const [expandedChunkId, setExpandedChunkId] = useState<string | null>(null);

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
          {message.restricted ? "Nothing in your cleared documents covers that." : message.content}
          {message.streaming && <span className={styles.cursor} aria-hidden="true" />}
        </p>
        {message.citations.length > 0 && (
          <>
            <div className={styles.citations}>
              {[...message.citations]
                .sort((a, b) => Number(isFullCitation(b) && b.cited) - Number(isFullCitation(a) && a.cited))
                .map((c, i) => {
                  const chunkId = typeof c === "string" ? c : c.chunk_id;
                  const isOpen = expandedChunkId === chunkId;
                  const isCited = isFullCitation(c) && c.cited;
                  return (
                    <button
                      key={chunkId}
                      type="button"
                      className={`${styles.citationTab} ${isOpen ? styles.citationTabOpen : ""} ${isCited ? styles.citationTabCited : ""}`}
                      aria-expanded={isOpen}
                      onClick={() => setExpandedChunkId(isOpen ? null : chunkId)}
                    >
                      {isCited && <span className={styles.citedBadge}>Cited</span>}
                      {citationLabel(c) || `Source ${i + 1}`}
                    </button>
                  );
                })}
            </div>
            {message.citations.map((c) => {
              const chunkId = typeof c === "string" ? c : c.chunk_id;
              if (chunkId !== expandedChunkId) return null;
              return (
                <div key={chunkId} className={styles.citationExcerpt}>
                  {isFullCitation(c) && c.text ? (
                    <p className={styles.citationExcerptText}>{c.text}</p>
                  ) : (
                    <p className={styles.citationExcerptMuted}>
                      Excerpt not available for this citation.
                    </p>
                  )}
                </div>
              );
            })}
          </>
        )}
      </div>
    </div>
  );
}
