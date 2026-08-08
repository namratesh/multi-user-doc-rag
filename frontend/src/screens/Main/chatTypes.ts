import type { Citation } from "../../api/types";

export interface UIMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[] | string[];
  restricted?: boolean;
  pending?: boolean;
}

export function isFullCitation(c: Citation | string): c is Citation {
  return typeof c !== "string";
}

export function citationLabel(c: Citation | string): string {
  if (!isFullCitation(c)) return c;
  const parts = [c.company_id];
  const period = [c.fiscal_quarter, c.fiscal_year].filter(Boolean).join(" ");
  if (period) parts.push(period);
  if (c.speaker_name) parts.push(c.speaker_name);
  return parts.filter(Boolean).join(" · ");
}
