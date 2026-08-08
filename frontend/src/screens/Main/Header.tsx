import { Wordmark } from "../../components/Wordmark/Wordmark";
import { Stamp } from "../../components/Stamp/Stamp";
import styles from "./Header.module.css";

interface HeaderProps {
  email: string;
  companies: string[];
  onSwitchReader: () => void;
}

export function Header({ email, companies, onSwitchReader }: HeaderProps) {
  return (
    <header className={styles.header}>
      <Wordmark onDark size="sm" />
      <div className={styles.right}>
        <div className={styles.identity}>
          <Stamp tone="granted" size="sm" lines={[companies.join(" · ")]} />
          <span className={styles.email}>{email}</span>
        </div>
        <button type="button" className={styles.switchLink} onClick={onSwitchReader}>
          Switch reader
        </button>
      </div>
    </header>
  );
}
