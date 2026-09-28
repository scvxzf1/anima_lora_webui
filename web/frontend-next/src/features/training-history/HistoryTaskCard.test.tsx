import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { HistoryTaskCard, formatHistoryTimestamp } from "./HistoryTaskCard";

afterEach(cleanup);

const props = {
  task: { id: "task-1", name: "Example", state: "completed", job: "training" },
  selected: [],
  busy: false,
  listSearch: "",
  onToggle: () => {},
};

it("formats epoch seconds and milliseconds while preserving the raw timestamp", () => {
  const seconds = 1_735_689_600;
  const milliseconds = seconds * 1000;
  const expected = formatHistoryTimestamp(seconds);
  expect(formatHistoryTimestamp(milliseconds)).toBe(expected);

  render(<MemoryRouter><HistoryTaskCard {...props} task={{ ...props.task, started_at: seconds }} /></MemoryRouter>);
  const time = screen.getByText(expected);
  expect(time.tagName).toBe("TIME");
  expect(time).toHaveAttribute("title", String(seconds));
  expect(time).toHaveAttribute("dateTime", new Date(milliseconds).toISOString());
});

it("falls back to saved text or a dash for missing and invalid timestamps", () => {
  expect(formatHistoryTimestamp("not-a-date")).toBe("not-a-date");
  expect(formatHistoryTimestamp(undefined)).toBe("—");
  render(<MemoryRouter><HistoryTaskCard {...props} task={{ ...props.task, started_at_text: "时间未记录" }} /></MemoryRouter>);
  expect(screen.getByText("时间未记录")).toHaveAttribute("title", "时间未记录");
});
