import { useEffect, useRef, useState } from 'react';
import { fetchMask, saveMask, type MaskImage } from './api';
import { loadBitmap, MaskCanvas } from './maskCanvas';

export function useMaskImage(file: string, index: number, image: string, onSaved: () => void) {
  const [engine, setEngine] = useState<MaskCanvas | null>(null);
  const [meta, setMeta] = useState<MaskImage | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [notice, setNotice] = useState('');
  const [reload, setReload] = useState(0);
  const [, changed] = useState(0);
  const pending = useRef(false);
  useEffect(() => {
    const controller = new AbortController();
    setEngine(null); setMeta(null); setError(''); setNotice('');
    if (!image) { setLoading(false); return; }
    setLoading(true);
    void (async () => {
      try {
        const result = await fetchMask(file, index, image, controller.signal);
        const original = await loadBitmap(result.image_url);
        const mask = await loadBitmap(result.mask_url);
        if (!controller.signal.aborted) { setEngine(new MaskCanvas(original, mask)); setMeta(result); }
      } catch (reason) {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : String(reason));
      } finally { if (!controller.signal.aborted) setLoading(false); }
    })();
    return () => controller.abort();
  }, [file, index, image, reload]);
  async function save() {
    if (!engine || !meta || pending.current || meta.readonly) return;
    pending.current = true; setSaving(true); setError(''); setNotice('');
    const version = engine.version;
    try {
      const result = await saveMask(file, index, image, meta.revision, await engine.toBlob());
      engine.markSaved(version); setMeta({ ...meta, revision: result.revision, has_mask: true });
      setNotice('蒙版已保存'); onSaved();
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { pending.current = false; setSaving(false); }
  }
  return { engine, meta, error, loading, saving, notice, save,
    changed: () => { changed(value => value + 1); setNotice(''); },
    reload: () => setReload(value => value + 1), dirty: engine?.dirty ?? false };
}
