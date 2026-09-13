import { expect, it } from "vitest";
import { promptLines, updatePromptLine, movePromptLine, promptRowError } from "./promptDocument";

it("preserves comments, blank lines, untouched prompts and unknown options", () => {
  const text = "# note\r\n\r\nfirst --w 512 --custom yes\r\n second --l 7  \r\n";
  const rows = promptLines(text);
  expect(rows).toHaveLength(2);
  const updated = updatePromptLine(text, rows[0].index, { ...rows[0].row, width: "768" });
  expect(updated).toBe("# note\r\n\r\nfirst --w 768 --custom yes\r\n second --l 7  \r\n");
  expect(movePromptLine("# header\na\n\nb", 1, 1)).toBe("# header\nb\n\na");
  expect(promptRowError({ ...rows[0].row, steps: "0" })).not.toBe("");
});
