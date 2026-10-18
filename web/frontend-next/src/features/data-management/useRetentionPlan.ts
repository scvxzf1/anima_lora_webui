import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { previewRetention } from "./retentionApi";
import type { RetentionPolicy } from "./retentionTypes";

const PREVIEW_DEBOUNCE_MS = 350;

export function useRetentionPlan({
  rootId,
  dirs,
  policy,
  excludeUnits,
  enabled,
}: {
  rootId: string;
  dirs: string[];
  policy: RetentionPolicy;
  excludeUnits: string[];
  enabled: boolean;
}) {
  const [debounced, setDebounced] = useState<RetentionPolicy>(policy);
  const policyKey = useMemo(() => JSON.stringify(policy), [policy]);
  const debouncedPolicyKey = useMemo(() => JSON.stringify(debounced), [debounced]);
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(policy), PREVIEW_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [policy]);

  const query = useQuery({
    queryKey: [
      "data-management",
      "retention",
      rootId,
      dirs.join("|"),
      debouncedPolicyKey,
      excludeUnits.join("|"),
    ],
    queryFn: ({ signal }) => previewRetention(rootId, dirs, debounced, excludeUnits, signal),
    enabled: enabled && Boolean(rootId) && dirs.length > 0,
    staleTime: 30_000,
    retry: false,
    placeholderData: keepPreviousData,
  });
  return { ...query, isSettling: policyKey !== debouncedPolicyKey };
}
