import { useState, type FormEvent } from "react";
import { Wordmark } from "../../components/Wordmark/Wordmark";
import { Button } from "../../components/Button/Button";
import { DEMO_USERS } from "./demoUsers";
import styles from "./Login.module.css";

interface LoginProps {
  onSubmit: (email: string) => Promise<void> | void;
  isLoggingIn: boolean;
  error: string | null;
}

export function Login({ onSubmit, isLoggingIn, error }: LoginProps) {
  const [email, setEmail] = useState("");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!email.trim() || isLoggingIn) return;
    void onSubmit(email);
  }

  function handleDemoClick(demoEmail: string) {
    if (isLoggingIn) return;
    setEmail(demoEmail);
    void onSubmit(demoEmail);
  }

  return (
    <div className={styles.page}>
      <div className={styles.brand}>
        <Wordmark />
      </div>

      <div className={styles.center}>
        <div className={styles.card}>
          <div className={styles.eyebrow}>Reading room · restricted archive</div>
          <h1 className={styles.title}>Request clearance</h1>
          <p className={styles.subtitle}>
            Sign in with your clearance email to open the desk. Access is scoped per
            reader &mdash; you'll only see documents you're authorized for.
          </p>

          <form onSubmit={handleSubmit}>
            <div className={styles.field}>
              <label className={styles.label} htmlFor="clearance-email">
                Enter your clearance email
              </label>
              <input
                id="clearance-email"
                className={styles.input}
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
                autoFocus
              />
            </div>

            {error && <div className={styles.error}>{error}</div>}

            <Button type="submit" block disabled={isLoggingIn || !email.trim()}>
              {isLoggingIn ? "Verifying…" : "Submit request"}
            </Button>
          </form>

          <div className={styles.divider}>
            <span className={styles.dividerLabel}>Or enter as</span>
          </div>

          <div className={styles.demoList}>
            {DEMO_USERS.map((user) => (
              <Button
                key={user.email}
                type="button"
                variant="ghost"
                className={styles.demoBtn}
                disabled={isLoggingIn}
                onClick={() => handleDemoClick(user.email)}
              >
                <span className={styles.demoEmail}>{user.email}</span>
                <span className={styles.demoCompanies}>{user.companies.join(" · ")}</span>
              </Button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
