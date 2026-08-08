export interface DemoUser {
  email: string;
  companies: string[];
}

export const DEMO_USERS: DemoUser[] = [
  { email: "alice@example.com", companies: ["TCS", "Infosys"] },
  { email: "bob@example.com", companies: ["Axis"] },
  { email: "carol@example.com", companies: ["Hdfc"] },
  { email: "dave@example.com", companies: ["TataTechnologies"] },
  { email: "eve@example.com", companies: ["TCS", "Hdfc"] },
];
