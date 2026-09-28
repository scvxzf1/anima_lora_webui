import { useNavigate, useSearchParams } from "react-router-dom";
import { HistoryTimeline } from "./HistoryTimeline";

export function HistoryTimelinePage() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const taskIds = params.getAll("task").filter(Boolean);
  const returnSearch = new URLSearchParams(params.get("from") || "").toString();

  return (
    <main className="history-page">
      <HistoryTimeline
        taskIds={taskIds}
        onClose={() => navigate(`/history${returnSearch ? `?${returnSearch}` : ""}`)}
      />
    </main>
  );
}
