import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { List } from "lucide-react";
import { NavLink, useLocation, useSearchParams } from "react-router-dom";
import { captioningKeys, fetchCaptionJobs, fetchCaptionProfiles } from "./api";
import { CaptionSource } from "./CaptionSource";
import { CaptionReview } from "./CaptionReview";
import { CaptionProviders } from "./CaptionProviders";
import { CaptionPrompts } from "./CaptionPrompts";
import { CaptionAssets } from "./CaptionAssets";
import { CaptionJobCleanup } from "./CaptionJobCleanup";
import "../settings/settings.css";
import "./captioning.css";

export function CaptioningPage() {
  const [params, setParams] = useSearchParams();
  const location = useLocation();
  const section = location.pathname.split("/").filter(Boolean).at(-1);
  const profiles = useQuery({
    queryKey: captioningKeys.profiles,
    queryFn: ({ signal }) => fetchCaptionProfiles(signal),
  });
  const jobs = useQuery({
    queryKey: captioningKeys.jobs,
    queryFn: ({ signal }) => fetchCaptionJobs(signal),
    refetchInterval: 5000,
  });
  const selectedJob = params.get("job") || "";
  const [showJobs, setShowJobs] = useState(false);
  return (
    <div className="caption-shell">
      <main className="caption-page">
        <header className="page-heading">
          <div>
            <p className="eyebrow">CAPTIONING</p>
            <h1>打标工作台</h1>
          </div>
        </header>
        <nav className="page-tabs" aria-label="打标视图">
          <NavLink to="/captioning" end>
            图片与任务
          </NavLink>
          <NavLink to="/captioning/providers">接入预设</NavLink>
          <NavLink to="/captioning/prompts">提示词预设</NavLink>
          <NavLink to="/captioning/assets">本地资源</NavLink>
        </nav>
        {profiles.error && (
          <p className="form-error" role="alert">
            {profiles.error.message}
          </p>
        )}
        {section === "assets" ? (
          <CaptionAssets />
        ) : section === "providers" ? (
          <CaptionProviders library={profiles.data} />
        ) : section === "prompts" ? (
          <CaptionPrompts />
        ) : (
          <>
            <button
              type="button"
              className="caption-jobs-toggle"
              aria-expanded={showJobs}
              aria-controls="caption-job-library"
              onClick={() => setShowJobs(!showJobs)}
            >
              <List size={16} />
              打标任务 ({jobs.data?.jobs.length ?? "..."})
            </button>
            <div className="caption-workspace" data-show-jobs={showJobs}>
              <aside className="object-library" id="caption-job-library">
                <header>
                  <h2>打标任务</h2>
                  <CaptionJobCleanup
                    jobs={jobs.data?.jobs || []}
                    onRemoved={(ids) => {
                      if (ids.includes(selectedJob)) setParams({});
                    }}
                  />
                  <button
                    type="button"
                    onClick={() => {
                      setParams({});
                      setShowJobs(false);
                    }}
                  >
                    新任务
                  </button>
                </header>
                {jobs.error && (
                  <p className="form-error" role="alert">
                    {jobs.error.message}
                  </p>
                )}
                {(jobs.data?.jobs || []).map((job) => (
                  <button
                    className="object-row"
                    type="button"
                    key={job.id}
                    data-selected={selectedJob === job.id}
                    onClick={() => {
                      setParams({ job: job.id });
                      setShowJobs(false);
                    }}
                  >
                    <span>
                      {job.profile_name} · {job.completed}/{job.total}
                    </span>
                    <small>
                      {job.state} · {job.created_at_text || job.id}
                    </small>
                  </button>
                ))}
                {jobs.data?.jobs.length === 0 && (
                  <p className="empty-state">暂无打标任务</p>
                )}
              </aside>
              <div className="caption-main">
                {selectedJob ? (
                  <CaptionReview key={selectedJob} jobId={selectedJob} />
                ) : (
                  <CaptionSource
                    library={profiles.data}
                    onCreated={(id) => setParams({ job: id })}
                  />
                )}
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
