import { useMutation } from '@tanstack/react-query';
import { Download } from 'lucide-react';

async function downloadDatasetGroup(groupId: string) {
  const query = new URLSearchParams({ kind: 'dataset' });
  const response = await fetch(
    `/api/config/file-groups/${encodeURIComponent(groupId)}/export?${query.toString()}`,
  );
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    const error = body && typeof body === 'object' && 'error' in body ? body.error : null;
    throw new Error(typeof error === 'string' && error ? error : `导出分组失败 (${response.status})`);
  }

  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement('a');
  link.href = url;
  link.download = 'dataset-group.zip';
  link.click();
  URL.revokeObjectURL(url);
}

export function DatasetGroupExport({
  groupId,
  label,
  empty,
}: {
  groupId: string;
  label: string;
  empty: boolean;
}) {
  const mutation = useMutation({ mutationFn: downloadDatasetGroup, retry: false });

  return (
    <>
      <button
        type="button"
        className="dataset-sort-button"
        aria-label={`导出分组 ${label}`}
        title="导出分组 ZIP"
        disabled={empty || mutation.isPending}
        onClick={() => mutation.mutate(groupId)}
      >
        <Download size={15} aria-hidden="true" />
      </button>
      {mutation.error ? (
        <p className="dataset-command-error" role="alert">{mutation.error.message}</p>
      ) : null}
    </>
  );
}
