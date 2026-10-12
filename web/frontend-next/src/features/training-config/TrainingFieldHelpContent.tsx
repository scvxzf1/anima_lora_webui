import type { FieldHelpV2 } from "./domain/field-help.js";
import { FIELD_LABEL_ZH } from "./domain/labels-options.js";

function HelpText({ paragraphs }: { paragraphs: string[] }) {
  return <>{paragraphs.filter(Boolean).map((text, index) => <p key={index}>{text}</p>)}</>;
}

function HelpSection({ title, paragraphs }: { title: string; paragraphs: string[] }) {
  if (!paragraphs.some(Boolean)) return null;
  return <section className="training-field-help-section"><h3>{title}</h3><HelpText paragraphs={paragraphs} /></section>;
}

export function displayHelpValue(value: unknown, labels?: Record<string, string>) {
  if (value === undefined || value === null) return "未设置";
  if (value === "") return "空字符串";
  if (typeof value === "boolean") return value ? "开启" : "关闭";
  if (typeof value === "object") return JSON.stringify(value);
  return labels?.[String(value)] || String(value);
}

export function TrainingFieldHelpContent({ help, value, options, optionLabels }: {
  help: FieldHelpV2;
  value: unknown;
  options?: string[];
  optionLabels?: Record<string, string>;
}) {
  const choices = help.options?.filter((option) => !options || options.includes(option.value));
  const hasDetails = choices?.some((option) => option.detail);
  return (
    <>
      <p className="training-field-help-summary">{help.summary}</p>
      <section className="training-field-help-recommendation">
        <h3>建议</h3><HelpText paragraphs={help.recommendation.split("\n")} />
      </section>
      {!!choices?.length && <section className="training-field-help-section">
        <h3>选项怎么选</h3>
        <table className="training-field-help-options">
          <thead><tr><th scope="col">选项</th><th scope="col">适用情况</th>{hasDetails && <th scope="col">说明</th>}</tr></thead>
          <tbody>{choices.map((option) => <tr key={option.value} data-current={String(value) === option.value || undefined}>
            <td>
              {displayHelpValue(option.value === "true" ? true : option.value === "false" ? false : option.value, optionLabels)}
              <code>{option.value}</code>
              {String(value) === option.value && <small>当前选择</small>}
            </td>
            <td>{option.when}</td>{hasDetails && <td>{option.detail}</td>}
          </tr>)}</tbody>
        </table>
      </section>}
      {help.guidance?.map((section) => <HelpSection key={section.title} {...section} />)}
      {!!help.cautions?.length && <section className="training-field-help-section">
        <h3>注意</h3><ul>{help.cautions.map((text, index) => <li key={index}>{text}</li>)}</ul>
      </section>}
      {!!help.advanced?.length && <details className="training-field-help-advanced">
        <summary>进阶</summary>
        {help.advanced.map((section) => <HelpSection key={section.title} {...section} />)}
      </details>}
      {!!help.relatedKeys?.length && <section className="training-field-help-related">
        <h3>相关参数</h3>
        <ul>{help.relatedKeys.map((key) => <li key={key}>{FIELD_LABEL_ZH[key] || key} <code>{key}</code></li>)}</ul>
      </section>}
    </>
  );
}
