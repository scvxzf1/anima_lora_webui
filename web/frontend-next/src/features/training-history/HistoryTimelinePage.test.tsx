import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createMemoryRouter, RouterProvider, useLocation } from "react-router-dom";
import { afterEach, expect, it, vi } from "vitest";
import { HistoryTimelinePage } from "./HistoryTimelinePage";

vi.mock("./HistoryTimeline", () => ({
  HistoryTimeline: ({ taskIds, onClose }: { taskIds: string[]; onClose: () => void }) => (
    <div>
      <output>{taskIds.join(",")}</output>
      <button type="button" onClick={onClose}>返回历史</button>
    </div>
  ),
}));

afterEach(cleanup);

it("restores ordered task IDs from a deep link and returns to history filters", () => {
  function HistoryLocation() {
    const location = useLocation();
    return <p>{location.pathname + location.search}</p>;
  }
  const router = createMemoryRouter([
    { path: "/history/aggregate", element: <HistoryTimelinePage /> },
    { path: "/history", element: <HistoryLocation /> },
  ], { initialEntries: ["/history/aggregate?task=second&task=first&from=q%3Dname"] });
  render(<RouterProvider router={router} />);
  expect(screen.getByText("second,first")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "返回历史" }));
  expect(screen.getByText("/history?q=name")).toBeInTheDocument();
});
