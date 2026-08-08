import type { Citation } from "../../api/types";

export interface UIMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[] | string[];
  restricted?: boolean;
  pending?: boolean;
  streaming?: boolean;
}

export function isFullCitation(c: Citation | string): c is Citation {
  return typeof c !== "string";
}

export function citationLabel(c: Citation | string): string {
  if (!isFullCitation(c)) return c;
  const parts: string[] = [];
  if (c.speaker_name) parts.push(`Speaker: ${c.speaker_name}`);
  if (c.company_id) parts.push(`Source: ${c.company_id}`);
  const period = [c.fiscal_quarter, c.fiscal_year ? `FY ${c.fiscal_year}` : null].filter(Boolean).join(" ");
  if (period) parts.push(period);
  return parts.join(" · ");
}
