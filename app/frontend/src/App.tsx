import { Route, Routes } from "react-router-dom";

import Landing from "./pages/Landing";
import Results from "./pages/Results";
import Upload from "./pages/Upload";

export default function App() {
  return (
    <div className="page">
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/results" element={<Results />} />
      </Routes>
    </div>
  );
}
