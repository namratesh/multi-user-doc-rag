import type { ButtonHTMLAttributes } from "react";
import styles from "./Button.module.css";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost";
  block?: boolean;
}

export function Button({
  variant = "primary",
  block = false,
  className = "",
  ...rest
}: ButtonProps) {
  const classes = [styles.btn, styles[variant], block ? styles.block : "", className]
    .filter(Boolean)
    .join(" ");
  return <button className={classes} {...rest} />;
}
