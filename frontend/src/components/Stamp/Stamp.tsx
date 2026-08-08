import styles from "./Stamp.module.css";

interface StampProps {
  tone: "granted" | "restricted";
  size?: "lg" | "sm" | "inline";
  lines: string[];
  animateKey?: string | number;
}

export function Stamp({ tone, size = "lg", lines, animateKey }: StampProps) {
  const classes = [
    styles.stamp,
    tone === "restricted" ? styles.restricted : "",
    styles[size],
    styles.animate,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={classes} key={animateKey} role="status">
      {lines.map((line) => (
        <span className={styles.label} key={line}>
          {line}
        </span>
      ))}
    </div>
  );
}
