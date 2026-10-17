import '@testing-library/jest-dom/vitest';

// Radix Switch measures its hidden bubble input in jsdom; jsdom does not ship
// ResizeObserver, so provide the smallest no-op implementation for UI tests.
if (typeof globalThis.ResizeObserver === 'undefined') {
  class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver;
}
