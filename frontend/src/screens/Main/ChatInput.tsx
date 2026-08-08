import { useState, type FormEvent, type KeyboardEvent } from "react";
import { Button } from "../../components/Button/Button";
import styles from "./ChatInput.module.css";

interface ChatInputProps {
  onSend: (text: string) => void;
  disabled: boolean;
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [value, setValue] = useState("");
  const [focused, setFocused] = useState(false);

  function submit(e?: FormEvent) {
    e?.preventDefault();
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue("");
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <div className={styles.bar}>
      <form className={styles.form} onSubmit={submit}>
        <div className={`${styles.inputWrap} ${focused ? styles.focused : ""}`}>
          <textarea
            className={styles.input}
            placeholder="Ask about a document you're cleared to see…"
            rows={1}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
          />
        </div>
        <Button type="submit" disabled={disabled || !value.trim()}>
          Ask
        </Button>
      </form>
    </div>
  );
}
