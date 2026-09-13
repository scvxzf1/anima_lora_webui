import { useMutation } from "@tanstack/react-query";
import { Languages } from "lucide-react";
import { useState } from "react";
import { apiRequest } from "../../api/client";

export function CaptionTranslation({
  value,
  disabled,
  onApply,
}: {
  value: string;
  disabled: boolean;
  onApply: (value: string) => void;
}) {
  const [language, setLanguage] = useState("zh");
  const translation = useMutation({
    mutationFn: async (source: string) => {
      const tags = source
        .split(/[,\n]/)
        .map((tag) => tag.trim())
        .filter(Boolean);
      if (!tags.length || tags.length > 500)
        throw new Error("单次翻译需要 1 至 500 个标签");
      const result = await apiRequest<{
        translations: string[];
        matched: number;
        total: number;
      }>("/api/captioning/translate-tags", {
        method: "POST",
        body: JSON.stringify({ tags, target_language: language }),
      });
      return { ...result, source };
    },
    retry: false,
  });
  const current = translation.data?.source === value;
  return (
    <details className="caption-translation">
      <summary>标签翻译</summary>
      <div className="toolbar">
        <select
          aria-label="翻译目标语言"
          value={language}
          disabled={translation.isPending}
          onChange={(e) => {
            setLanguage(e.target.value);
            translation.reset();
          }}
        >
          <option value="zh">中文</option>
          <option value="en">英文</option>
        </select>
        <button
          type="button"
          disabled={!value.trim() || disabled || translation.isPending}
          onClick={() => translation.mutate(value)}
        >
          <Languages size={16} />
          查询词典
        </button>
      </div>
      {translation.error && (
        <p role="alert" className="form-error">
          {translation.error.message}
        </p>
      )}
      {translation.data && (
        <>
          <p className="caption-translation-result">
            {translation.data.translations.join(", ")}
          </p>
          <div className="toolbar">
            <small>
              匹配 {translation.data.matched} / {translation.data.total}
            </small>
            <button
              type="button"
              disabled={!current || disabled}
              onClick={() => onApply(translation.data!.translations.join(", "))}
            >
              采用翻译候选
            </button>
          </div>
        </>
      )}
    </details>
  );
}
