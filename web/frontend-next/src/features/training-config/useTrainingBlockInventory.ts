import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "../../api/client";

export type ModelBlockInventory = { block_count: number; max_swap: number };

export function useTrainingBlockInventory(family: string, path: string) {
  const identity = `${family}\n${path}`;
  const [settled, setSettled] = useState(identity);
  useEffect(() => {
    const timer = window.setTimeout(() => setSettled(identity), 300);
    return () => window.clearTimeout(timer);
  }, [identity]);
  const ready = settled === identity;
  const query = useQuery({
    queryKey: ["training-model-blocks", family, path],
    enabled: Boolean(path) && ready,
    queryFn: ({ signal }) => apiRequest<ModelBlockInventory>(`/api/config/model-blocks?${new URLSearchParams({
      model_family: family, pretrained_model_name_or_path: path,
    })}`, { signal }),
    staleTime: 30_000,
    // Layout/stage switches reuse the current header; explicit refresh still re-reads it.
    refetchOnMount: false,
    retry: false,
  });
  // Cached data must not constrain a changed model or a pending/failed refresh.
  const inventory = ready && !query.isFetching && !query.isError ? query.data : undefined;
  return { ready, query, inventory };
}

export type TrainingBlockLookup = ReturnType<typeof useTrainingBlockInventory>;
