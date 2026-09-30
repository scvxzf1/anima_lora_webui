import { Copy, FolderOpen, RefreshCw } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useWatch, type FieldPath, type UseFormReturn } from 'react-hook-form';

import { apiRequest } from '../../api/client';
import type { DatasetFormValues } from './datasetForm';
import './DatasetSourcePathField.css';

type Props = {
  form: UseFormReturn<DatasetFormValues>;
  path: FieldPath<DatasetFormValues>;
  label: string;
};

type Inspection = {
  ok?: boolean;
  error?: string;
  source_exists?: boolean;
  source_is_dir?: boolean;
  source_image_count?: number;
  source_inspection_error?: string | null;
};

export function DatasetSourcePathField({ form, path, label }: Props) {
  const value = useWatch({ control: form.control, name: path });
  const inputRef = useRef<HTMLInputElement>(null);
  const requestId = useRef(0);
  const activeRequest = useRef<AbortController | null>(null);
  const debounceTimer = useRef<number | null>(null);
  const fallbackAbort = useRef<AbortController | null>(null);
  const firstCheck = useRef(true);
  const [status, setStatus] = useState('');
  const error = form.getFieldState(path, form.formState).error;
  const errorId = `${path.replaceAll('.', '-')}-error`;
  const statusId = `${path.replaceAll('.', '-')}-status`;
  const errorMessage = typeof error?.message === 'string' ? error.message : '';
  const registration = form.register(path);

  useEffect(() => () => {
    if (debounceTimer.current !== null) window.clearTimeout(debounceTimer.current);
    activeRequest.current?.abort();
    fallbackAbort.current?.abort();
  }, []);

  useEffect(() => {
    scheduleInspection(String(value ?? '').trim());
    return () => {
      if (debounceTimer.current !== null) window.clearTimeout(debounceTimer.current);
      debounceTimer.current = null;
      requestId.current += 1;
      activeRequest.current?.abort();
      activeRequest.current = null;
    };
  }, [path, value]);

  function inspect(source: string) {
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    const id = ++requestId.current;
    const query = new URLSearchParams({ source_image_dir: source, inspect: '1' });
    void apiRequest<Inspection>(`/api/config/data-dirs/suggest?${query}`, { signal: controller.signal })
      .then((result) => {
        if (requestId.current === id && !controller.signal.aborted) setStatus(describeInspection(result));
      })
      .catch((cause: unknown) => {
        if (requestId.current === id && !controller.signal.aborted) setStatus(cause instanceof Error ? cause.message : '路径检测失败');
      });
  }

  function scheduleInspection(source: string) {
    if (debounceTimer.current !== null) window.clearTimeout(debounceTimer.current);
    debounceTimer.current = null;
    requestId.current += 1;
    activeRequest.current?.abort();
    activeRequest.current = null;
    setStatus(source ? '正在检测…' : '请输入路径');
    if (!source) return;
    const delay = firstCheck.current ? 80 : 420;
    firstCheck.current = false;
    debounceTimer.current = window.setTimeout(() => {
      debounceTimer.current = null;
      inspect(source);
    }, delay);
  }

  function refresh() {
    const source = String(form.getValues(path) ?? '').trim();
    if (!source) {
      setStatus('请输入路径');
      return;
    }
    if (debounceTimer.current !== null) window.clearTimeout(debounceTimer.current);
    debounceTimer.current = null;
    requestId.current += 1;
    activeRequest.current?.abort();
    setStatus('正在检测…');
    firstCheck.current = false;
    inspect(source);
  }

  async function copyPath() {
    const input = inputRef.current;
    const source = String(form.getValues(path) ?? '').trim();
    if (!source || !input) return;
    try {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(source);
      else {
        input.select();
        if (!document.execCommand('copy')) throw new Error('copy failed');
        input.setSelectionRange(source.length, source.length);
      }
      setStatus('目录路径已复制');
    } catch {
      setStatus('无法访问剪贴板，请手动复制路径');
    }
  }

  async function chooseDirectory() {
    const fallbackController = new AbortController();
    fallbackAbort.current = fallbackController;
    try {
      const picker = (window as Window & { showDirectoryPicker?: (options: { mode: 'read' }) => Promise<{ name: string }> }).showDirectoryPicker;
      const name = picker ? (await picker({ mode: 'read' })).name : await chooseDirectoryFallback(fallbackController.signal);
      if (!name) return;
      form.setValue(path, name as never, { shouldDirty: true, shouldValidate: true });
      setStatus('已填入所选目录名；绝对路径可直接粘贴');
    } catch (cause) {
      if (!fallbackController.signal.aborted && !(cause instanceof DOMException && cause.name === 'AbortError')) setStatus('无法读取所选目录');
    } finally {
      if (fallbackAbort.current === fallbackController) fallbackAbort.current = null;
    }
  }

  return (
    <div className="dataset-source-path-field dataset-wide-field">
      <label htmlFor={path}>
        <span>{label}</span>
        <input
          id={path}
          {...registration}
          ref={(element) => { registration.ref(element); inputRef.current = element; }}
          aria-invalid={errorMessage ? true : undefined}
          aria-describedby={[errorMessage ? errorId : '', statusId].filter(Boolean).join(' ') || undefined}
        />
      </label>
      <div className="dataset-source-path-actions">
        <button type="button" title="选择本机文件夹名" aria-label="选择本机文件夹名" onClick={() => void chooseDirectory()}>
          <FolderOpen size={15} aria-hidden="true" />
        </button>
        <button type="button" title="复制目录路径" aria-label="复制目录路径" onClick={() => void copyPath()}>
          <Copy size={15} aria-hidden="true" />
        </button>
        <button type="button" title="重新检查目录" aria-label="重新检查目录" onClick={refresh}>
          <RefreshCw size={15} aria-hidden="true" />
        </button>
      </div>
      {errorMessage && <small id={errorId} role="alert">{errorMessage}</small>}
      <small id={statusId} aria-live="polite" data-state={status.startsWith('检测到') ? 'valid' : status ? 'checking' : 'idle'}>{status}</small>
    </div>
  );
}

function describeInspection(result: Inspection) {
  if (result.ok === false) return result.error || '路径检测失败';
  if (typeof result.source_exists !== 'boolean') return '重启服务后检测';
  if (result.source_inspection_error) return '目录无法完整读取';
  if (!result.source_exists) return '路径不存在';
  if (!result.source_is_dir) return '路径不是目录';
  const count = Number(result.source_image_count || 0);
  return count > 0 ? `检测到 ${count} 张图片` : '目录存在，未检测到图片';
}

function chooseDirectoryFallback(signal: AbortSignal): Promise<string> {
  return new Promise((resolve) => {
    const picker = document.createElement('input');
    picker.type = 'file';
    picker.multiple = true;
    picker.setAttribute('webkitdirectory', '');
    picker.hidden = true;
    const finish = (relative = '') => {
      if (!picker.isConnected) return;
      signal.removeEventListener('abort', onAbort);
      picker.remove();
      resolve(relative.split('/').filter(Boolean)[0] || '');
    };
    const onAbort = () => finish();
    picker.addEventListener('change', () => {
      const relative = (picker.files?.[0] as (File & { webkitRelativePath?: string }) | undefined)?.webkitRelativePath || '';
      finish(relative);
    }, { once: true });
    picker.addEventListener('cancel', () => finish(), { once: true });
    signal.addEventListener('abort', onAbort, { once: true });
    document.body.appendChild(picker);
    picker.click();
  });
}
