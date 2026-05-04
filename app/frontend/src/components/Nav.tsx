import { Link, useNavigate } from "react-router-dom";

import ThemeToggle from "./ThemeToggle";

interface NavProps {
  /** Tab-layout per FR-06. Active tab is highlighted. */
  active?: "home" | "upload" | "results";
}

export default function Nav({ active = "home" }: NavProps) {
  const navigate = useNavigate();
  const tabStyle = (key: NavProps["active"]) => ({
    color: active === key ? "var(--c-text)" : "var(--c-muted)",
    fontWeight: active === key ? 500 : 400,
  });
  return (
    <nav className="nav">
      <div className="brand">
        <div className="brand-mark" />
        <div className="brand-name">
          pump.detect <span>/ v1.0</span>
        </div>
      </div>
      <div className="nav-mid">
        <Link to="/" style={tabStyle("home")}>
          Home
        </Link>
        <Link to="/upload" style={tabStyle("upload")}>
          Upload
        </Link>
        <Link to="/results" style={tabStyle("results")}>
          Results
        </Link>
      </div>
      <ThemeToggle />
      <button className="nav-cta" onClick={() => navigate("/upload")}>
        Get started
      </button>
    </nav>
  );
}
