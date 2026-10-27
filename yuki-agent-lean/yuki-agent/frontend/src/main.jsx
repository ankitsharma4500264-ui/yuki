import React from "react";
import { createRoot } from "react-dom/client";
import AgentPanel from "./yuki-agent/AgentPanel";

// VITE_API_BASE = your backend URL (direct mode). Leave empty to use same-origin /api (Vite dev proxy or a Vercel rewrite).
createRoot(document.getElementById("root")).render(
  <div style={{ height: "100vh" }}><AgentPanel apiBase={import.meta.env.VITE_API_BASE || ""} /></div>
);
