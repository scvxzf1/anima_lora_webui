import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';
import { useQueryClient } from '@tanstack/react-query';

import { datasetKeys, fetchDatasetPresetImages } from './api';
import { DATASET_PREVIEW_PAGE_SIZE, hasDatasetPreviewPagination } from './DatasetPreviewPager';
import type { DatasetPreviewResponse } from './types';

const MAX_PAGES = 5;
type Direction = 'before' | 'after';
type LoadError = { direction: Direction | 'refresh'; message: string };
type WindowState = { key: string; pages: DatasetPreviewResponse[]; visibleOffset: number };

export function useContinuousPreview({ enabled, identity, offset, data, bodyRef }: {
  enabled: boolean;
  identity: string;
  offset: number;
  data?: DatasetPreviewResponse;
  bodyRef: RefObject<HTMLDivElement | null>;
}) {
  const key = `${identity}\0${offset}`;
  const queryClient = useQueryClient();
  const [windowState, setWindowState] = useState<WindowState>({ key, pages: [], visibleOffset: offset });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<LoadError | null>(null);
  const controller = useRef<AbortController | null>(null);
  const retryTimer = useRef<number | null>(null);
  const anchor = useRef<{ offset: number; top: number } | null>(null);
  const generation = useRef(0);
  const active = enabled && !!data && hasDatasetPreviewPagination(data);
  const pages = active && windowState.key === key ? windowState.pages : [];
  const visibleOffset = pages.length ? windowState.visibleOffset : offset;

  useEffect(() => {
    generation.current += 1;
    controller.current?.abort();
    controller.current = null;
    if (retryTimer.current !== null) window.clearTimeout(retryTimer.current);
    retryTimer.current = null;
    setLoading(false);
    setError(null);
    setWindowState({ key, pages: active && data ? [data] : [], visibleOffset: offset });
    return () => {
      generation.current += 1;
      controller.current?.abort();
      controller.current = null;
      if (retryTimer.current !== null) window.clearTimeout(retryTimer.current);
      retryTimer.current = null;
    };
  }, [key, active]);

  useEffect(() => {
    if (!active || !data || windowState.key !== key) return;
    if (windowState.pages.length && windowState.pages[0].total !== data.total) {
      generation.current += 1;
      controller.current?.abort();
      controller.current = null;
      setLoading(false);
      setError(null);
    }
    setWindowState((previous) => {
      if (previous.key !== key || !previous.pages.length) return previous;
      if (previous.pages[0].total !== data.total) {
        return { key, pages: [data], visibleOffset: offset };
      }
      const index = previous.pages.findIndex((page) => page.offset === data.offset);
      if (index < 0 || previous.pages[index] === data) return previous;
      const updated = [...previous.pages];
      updated[index] = data;
      return { ...previous, pages: updated };
    });
  }, [active, data, key, offset, windowState.key]);

  const refresh = async () => {
    if (!active || !pages.length || document.visibilityState === 'hidden') return;
    generation.current += 1;
    controller.current?.abort();
    const request = new AbortController();
    controller.current = request;
    const requestGeneration = generation.current;
    setLoading(true);
    setError(null);
    const [file, datasetIndex] = identity.split('\0');
    try {
      const offsets = [...new Set([...pages.map((page) => page.offset!), offset])];
      const results = await Promise.all(offsets.map((pageOffset) =>
        fetchDatasetPresetImages(file, Number(datasetIndex), request.signal, DATASET_PREVIEW_PAGE_SIZE, pageOffset)));
      if (request.signal.aborted || requestGeneration !== generation.current) return;
      if (results.some((page) => !hasDatasetPreviewPagination(page))) throw new Error('分页接口尚未生效');
      const refreshed = [...new Map(results.slice(0, pages.length).map((page) => [page.offset, page])).values()];
      const body = bodyRef.current;
      const anchorPage = body?.querySelector<HTMLElement>(`[data-preview-offset="${visibleOffset}"]`);
      if (anchorPage) anchor.current = { offset: visibleOffset, top: anchorPage.getBoundingClientRect().top };
      setWindowState({ key, pages: refreshed, visibleOffset: refreshed.some((page) => page.offset === visibleOffset) ? visibleOffset : refreshed[refreshed.length - 1].offset! });
      queryClient.setQueryData(datasetKeys.preview(file, Number(datasetIndex), offset), results[offsets.indexOf(offset)]);
    } catch (failure) {
      if (!request.signal.aborted && requestGeneration === generation.current) {
        setError({ direction: 'refresh', message: failure instanceof Error ? failure.message : '无法刷新图片预览' });
      }
    } finally {
      if (controller.current === request && requestGeneration === generation.current) {
        controller.current = null;
        setLoading(false);
      }
    }
  };
  const refreshRef = useRef(refresh);
  refreshRef.current = refresh;

  useLayoutEffect(() => {
    const saved = anchor.current;
    anchor.current = null;
    const body = bodyRef.current;
    if (!saved || !body) return;
    const page = Array.from(body.querySelectorAll<HTMLElement>('[data-preview-offset]'))
      .find((element) => Number(element.dataset.previewOffset) === saved.offset);
    if (page) body.scrollTop += page.getBoundingClientRect().top - saved.top;
  }, [bodyRef, pages]);

  const load = useCallback(async (direction: Direction) => {
    if (!active || !pages.length || controller.current || error || document.visibilityState === 'hidden') return;
    const boundary = direction === 'before' ? pages[0] : pages[pages.length - 1];
    if (direction === 'before' ? !boundary.has_more_before : !boundary.has_more_after) return;
    const nextOffset = direction === 'before'
      ? Math.max(0, boundary.offset! - boundary.limit)
      : boundary.next_offset ?? boundary.offset! + boundary.images.length;
    if (pages.some((page) => page.offset === nextOffset) || nextOffset === boundary.offset) return;
    const request = new AbortController();
    controller.current = request;
    const requestGeneration = generation.current;
    setLoading(true);
    try {
      const result = await fetchDatasetPresetImages(identity.split('\0')[0], Number(identity.split('\0')[1]), request.signal, DATASET_PREVIEW_PAGE_SIZE, nextOffset);
      if (request.signal.aborted || requestGeneration !== generation.current) return;
      if (!hasDatasetPreviewPagination(result) || result.offset !== nextOffset) throw new Error('分页结果与请求位置不一致');
      const body = bodyRef.current;
      const anchorPage = body?.querySelector<HTMLElement>(`[data-preview-offset="${boundary.offset}"]`);
      if (anchorPage) anchor.current = { offset: boundary.offset!, top: anchorPage.getBoundingClientRect().top };
      setWindowState((previous) => {
        if (previous.key !== key || previous.pages.some((page) => page.offset === nextOffset)) return previous;
        const merged = direction === 'before' ? [result, ...previous.pages] : [...previous.pages, result];
        const bounded = direction === 'before' ? merged.slice(0, MAX_PAGES) : merged.slice(-MAX_PAGES);
        const visibleOffset = bounded.some((page) => page.offset === previous.visibleOffset)
          ? previous.visibleOffset : boundary.offset!;
        return { ...previous, pages: bounded, visibleOffset };
      });
    } catch (failure) {
      if (!request.signal.aborted && requestGeneration === generation.current) {
        setError({ direction, message: failure instanceof Error ? failure.message : '无法读取相邻页' });
      }
    } finally {
      if (controller.current === request && requestGeneration === generation.current) {
        controller.current = null;
        setLoading(false);
      }
    }
  }, [active, bodyRef, error, identity, key, pages]);

  const loadRef = useRef(load);
  loadRef.current = load;
  useEffect(() => {
    const body = bodyRef.current;
    if (!active || !body || !pages.length) return;
    let scrolled = false;
    let previousTop = body.scrollTop;
    const update = () => {
      const elements = Array.from(body.querySelectorAll<HTMLElement>('[data-preview-offset]'));
      const bodyTop = body.getBoundingClientRect().top;
      const visible = elements.find((element) => element.getBoundingClientRect().bottom > bodyTop + 56);
      if (visible) {
        const next = Number(visible.dataset.previewOffset);
        setWindowState((state) => state.key === key && state.visibleOffset !== next ? { ...state, visibleOffset: next } : state);
      }
      if (!scrolled || body.scrollHeight <= body.clientHeight) return;
      const goingUp = body.scrollTop < previousTop;
      previousTop = body.scrollTop;
      if (goingUp && body.scrollTop < 240) void loadRef.current('before');
      else if (!goingUp && body.scrollHeight - body.scrollTop - body.clientHeight < 240) void loadRef.current('after');
    };
    const onScroll = () => { scrolled = true; update(); };
    const onWheel = (event: WheelEvent) => {
      if (event.deltaY < 0 && body.scrollTop <= 0) void loadRef.current('before');
      else if (event.deltaY > 0 && body.scrollHeight - body.scrollTop - body.clientHeight <= 1) void loadRef.current('after');
    };
    body.addEventListener('scroll', onScroll, { passive: true });
    body.addEventListener('wheel', onWheel, { passive: true });
    const observer = typeof IntersectionObserver === 'undefined' ? null : new IntersectionObserver(() => { if (scrolled) update(); }, { root: body, threshold: [0, 0.5] });
    for (const element of body.querySelectorAll('[data-preview-offset]')) observer?.observe(element);
    return () => { body.removeEventListener('scroll', onScroll); body.removeEventListener('wheel', onWheel); observer?.disconnect(); };
  }, [active, bodyRef, key, pages]);

  return {
    pages,
    visibleOffset,
    loading,
    error,
    refresh,
    retry: () => { if (error && retryTimer.current === null) { const direction = error.direction; setError(null); retryTimer.current = window.setTimeout(() => { retryTimer.current = null; if (direction === 'refresh') void refreshRef.current(); else void loadRef.current(direction); }, 0); } },
  };
}
