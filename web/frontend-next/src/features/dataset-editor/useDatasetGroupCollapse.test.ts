import { act, cleanup, renderHook } from '@testing-library/react';
import { afterEach, expect, test, vi } from 'vitest';
import { DATASET_COLLAPSE_KEY, useDatasetGroupCollapse } from './useDatasetGroupCollapse';

afterEach(() => { cleanup(); localStorage.clear(); vi.useRealTimers(); vi.restoreAllMocks(); });

test('manual collapse persists, invalid storage is ignored and search does not overwrite it', () => {
  localStorage.setItem(DATASET_COLLAPSE_KEY, '{bad');
  const { result, rerender } = renderHook(({ search }) => useDatasetGroupCollapse(false, search), { initialProps: { search: false } });
  expect(result.current.isCollapsed('a')).toBe(false);
  act(() => result.current.toggle('a'));
  expect(JSON.parse(localStorage.getItem(DATASET_COLLAPSE_KEY)!)).toEqual(['a']);
  rerender({ search: true });
  expect(result.current.isCollapsed('a')).toBe(false);
  rerender({ search: false });
  expect(result.current.isCollapsed('a')).toBe(true);
});

test('dragging never changes the persisted collapsed state', () => {
  vi.useFakeTimers();
  localStorage.setItem(DATASET_COLLAPSE_KEY, '["a"]');
  const element = document.createElement('section');
  element.className = 'dataset-group';
  element.dataset.groupId = 'a';
  const { result } = renderHook(() => useDatasetGroupCollapse(true, false, 'a'));
  act(() => result.current.toggle('a'));
  expect(result.current.isCollapsed('a')).toBe(true);
  expect(localStorage.getItem(DATASET_COLLAPSE_KEY)).toBe('["a"]');
});
