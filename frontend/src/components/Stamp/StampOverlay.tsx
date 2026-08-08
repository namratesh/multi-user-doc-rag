import { Stamp } from "./Stamp";
import styles from "./StampOverlay.module.css";

interface StampOverlayProps {
  email: string;
  companies: string[];
}

export function StampOverlay({ email, companies }: StampOverlayProps) {
  return (
    <div className={styles.overlay}>
      <div className={styles.stack}>
        <Stamp
          tone="granted"
          size="lg"
          lines={["Access granted", companies.join(" · ").toUpperCase()]}
          animateKey={email}
        />
        <span className={styles.caption}>{email}</span>
      </div>
    </div>
  );
}
