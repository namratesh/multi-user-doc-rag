export interface DemoUser {
  email: string;
  companies: string[];
}

export const DEMO_USERS: DemoUser[] = [
  { email: "alice@example.com", companies: ["TCS", "Infosys"] },
  { email: "bob@example.com", companies: ["Axis"] },
  { email: "carol@example.com", companies: ["Hdfc"] },
];
