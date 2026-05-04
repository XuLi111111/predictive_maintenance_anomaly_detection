import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";

import App from "./App";
import "./styles/tokens.css";
import "./styles/app.css";

const saved = localStorage.getItem("theme");
if (saved) document.documentElement.setAttribute("data-theme", saved);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
