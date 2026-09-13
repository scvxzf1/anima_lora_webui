export const datasetMaskModes = ['none', 'external', 'embedded', 'auto'] as const;
export type DatasetMaskMode = typeof datasetMaskModes[number];

export function datasetMaskFromRow(row: Record<string, unknown>) {
  const mask_dir = String(row.mask_dir || '').trim();
  const aliases: Record<string, string> = {
    off: 'none', disabled: 'none', alpha: 'embedded', image_alpha: 'embedded', mask_dir: 'external',
  };
  const raw = String(row.mask_mode || '').trim().toLowerCase().replaceAll('-', '_');
  let mask_mode = aliases[raw] || raw;
  if (!mask_mode || mask_mode === 'auto') {
    mask_mode = mask_dir ? 'external' : row.alpha_mask ? 'embedded' : 'auto';
  }
  // Unknown modes stay invalid so saving cannot silently change their semantics.
  return { mask_mode: mask_mode as DatasetMaskMode, mask_dir };
}

export function datasetMaskForWrite(row: { mask_mode: DatasetMaskMode; mask_dir: string }) {
  return {
    mask_mode: row.mask_mode,
    mask_dir: row.mask_mode === 'external' ? row.mask_dir.trim() : '',
    alpha_mask: row.mask_mode === 'external' || row.mask_mode === 'embedded',
  };
}
