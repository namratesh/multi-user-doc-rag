import styles from "./Wordmark.module.css";

interface WordmarkProps {
  onDark?: boolean;
  size?: "md" | "sm";
}

export function Wordmark({ onDark = false, size = "md" }: WordmarkProps) {
  const color = onDark ? "#eaede7" : "#141b2d";
  return (
    <div className={`${styles.wordmark} ${size === "sm" ? styles.sm : ""}`}>
      <svg viewBox="0 0 24 24" className={styles.mark} aria-hidden="true">
        <path
          d="M4 3h10l4 4v14H4V3z"
          fill="none"
          stroke={color}
          strokeWidth="1.6"
          strokeLinejoin="round"
        />
        <path d="M14 3l4 4h-4V3z" fill={color} />
      </svg>
      <span className={styles.word} style={{ color }}>
        Folio
      </span>
    </div>
  );
}
