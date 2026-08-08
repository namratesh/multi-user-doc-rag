import { useCallback, useState } from "react";
import { useAuth, type Session } from "./auth/useAuth";
import { Login } from "./screens/Login/Login";
import { Main } from "./screens/Main/Main";
import { StampOverlay } from "./components/Stamp/StampOverlay";

const STAMP_HOLD_MS = 900;

function App() {
  const { session, login, switchReader, isLoggingIn, error } = useAuth();
  const [pendingStamp, setPendingStamp] = useState<Session | null>(null);

  const handleLogin = useCallback(
    async (email: string) => {
      const next = await login(email);
      setPendingStamp(next);
      window.setTimeout(() => setPendingStamp(null), STAMP_HOLD_MS);
    },
    [login],
  );

  return (
    <>
      {session ? (
        <Main session={session} onSwitchReader={switchReader} />
      ) : (
        <Login onSubmit={handleLogin} isLoggingIn={isLoggingIn} error={error} />
      )}
      {pendingStamp && (
        <StampOverlay email={pendingStamp.email} companies={pendingStamp.companies} />
      )}
    </>
  );
}

export default App;
