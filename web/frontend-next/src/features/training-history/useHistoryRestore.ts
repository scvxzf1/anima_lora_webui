import { useEffect, useState } from "react";

type PagesQuery = {
  data?: { pages: unknown[] };
  hasNextPage: boolean;
  isFetching: boolean;
  error: Error | null;
  fetchNextPage: () => unknown;
};

export function historyReturnSearch(search: string, depth: number, anchor?: string) {
  const next = new URLSearchParams(search);
  next.set("depth", String(depth));
  if (anchor) next.set("anchor", anchor);
  return next.toString();
}

export function historyReturnDepth(raw: string | null) {
  const number = Number(raw);
  return Number.isSafeInteger(number) && number > 0 ? number : 1;
}

export function useHistoryRestore(query: PagesQuery, search: string) {
  const target = historyReturnDepth(new URLSearchParams(search).get("depth"));
  const loaded = query.data?.pages.length || 0;
  const [batch, setBatch] = useState({ search, until: 10 });
  const until = batch.search === search ? batch.until : 10;
  const restoring = loaded < target && (!query.data || query.hasNextPage);
  const paused = restoring && loaded >= until;
  useEffect(() => {
    if (restoring && !paused && query.data && !query.isFetching && !query.error) {
      void query.fetchNextPage();
    }
  }, [restoring, paused, loaded, query.data, query.isFetching, query.error, query.fetchNextPage]);
  return {
    loaded, target, restoring, paused,
    continueRestore: () => setBatch({ search, until: loaded + 10 }),
  };
}
