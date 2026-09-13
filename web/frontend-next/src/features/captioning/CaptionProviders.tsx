import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Plus, Pencil, Trash2, Check, Plug } from "lucide-react";
import { useState } from "react";
import {
  activateCaptionProfile,
  captioningKeys,
  deleteCaptionProfile,
  testCaptionProvider,
  type CaptionProfile,
  type Profiles,
} from "./api";
import { CaptionProfileEditor } from "./CaptionProfileEditor";

export function CaptionProviders({ library }: { library?: Profiles }) {
  const qc = useQueryClient();
  const [editing, setEditing] = useState<CaptionProfile | "new" | null>(null);
  const [notice, setNotice] = useState("");
  const action = useMutation({
    mutationFn: async ({
      id,
      mode,
    }: {
      id: string;
      mode: "activate" | "delete" | "test";
    }) => {
      if (mode === "activate") await activateCaptionProfile(id);
      else if (mode === "delete") await deleteCaptionProfile(id);
      else await testCaptionProvider(id);
      return mode;
    },
    retry: false,
    onSuccess: (mode) => {
      setNotice(mode === "test" ? "连接测试完成" : "接入预设已更新");
      qc.invalidateQueries({ queryKey: captioningKeys.profiles });
    },
  });
  return (
    <section className="settings-section">
      <header className="review-heading">
        <h2>接入预设</h2>
        <button
          type="button"
          onClick={() => setEditing("new")}
          disabled={action.isPending}
        >
          <Plus size={16} />
          新建接入
        </button>
      </header>
      <div className="provider-list">
        {library?.profiles.map((profile) => (
          <article key={profile.id}>
            <div>
              <h3>
                {profile.name}
                {library.active_profile_id === profile.id ? " · 当前" : ""}
              </h3>
              <p>
                {profile.provider} · {profile.status}
              </p>
              <small>
                {String(
                  profile.config.base_url || profile.config.asset_id || "",
                )}
              </small>
            </div>
            <div className="toolbar">
              <button
                type="button"
                disabled={action.isPending}
                onClick={() => setEditing(profile)}
              >
                <Pencil size={15} />
                编辑
              </button>
              <button
                type="button"
                disabled={
                  action.isPending ||
                  !profile.available ||
                  library.active_profile_id === profile.id
                }
                onClick={() =>
                  action.mutate({ id: profile.id, mode: "activate" })
                }
              >
                <Check size={15} />
                设为当前
              </button>
              <button
                type="button"
                disabled={action.isPending}
                onClick={() => {
                  if (
                    window.confirm(
                      `将测试 ${profile.name} 的连接，不提交图片。确认继续吗？`,
                    )
                  )
                    action.mutate({ id: profile.id, mode: "test" });
                }}
              >
                <Plug size={15} />
                测试
              </button>
              <button
                type="button"
                className="danger"
                disabled={action.isPending}
                onClick={() => {
                  if (window.confirm(`删除接入预设“${profile.name}”及其凭据？`))
                    action.mutate({ id: profile.id, mode: "delete" });
                }}
              >
                <Trash2 size={15} />
                删除
              </button>
            </div>
          </article>
        ))}
      </div>
      {!library?.profiles.length && <p>暂无接入预设</p>}
      {action.error && (
        <p role="alert" className="form-error">
          {action.error.message}
        </p>
      )}
      {notice && <p role="status">{notice}</p>}
      {editing && (
        <CaptionProfileEditor
          profile={editing === "new" ? undefined : editing}
          types={library?.provider_types || []}
          onClose={() => setEditing(null)}
        />
      )}
    </section>
  );
}
