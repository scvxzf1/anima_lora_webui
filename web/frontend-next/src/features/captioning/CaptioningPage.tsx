import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { List } from "lucide-react";
import { NavLink, useLocation, useSearchParams } from "react-router-dom";
import { QueryFeedback } from "../../components/QueryFeedback";
import { captioningKeys, fetchCaptionJobs, fetchCaptionProfiles } from "./api";
import { CaptionSource } from "./CaptionSource";
import { CaptionReview } from "./CaptionReview";
import { CaptionProviders } from "./CaptionProviders";
import { CaptionPrompts } from "./CaptionPrompts";
import { CaptionAssets } from "./CaptionAssets";
import { CaptionJobCleanup } from "./CaptionJobCleanup";
import "../settings/settings.css";
import "./captioning.css";

export function CaptioningPage({ embeddedBasePath = "/captioning" }: { embeddedBasePath?: string }) {
  const [params, setParams] = useSearchParams();
  const location = useLocation();
  const section = location.pathname.split("/").filter(Boolean).at(-1);
  const profiles = useQuery({
    queryKey: captioningKeys.profiles,
    queryFn: ({ signal }) => fetchCaptionProfiles(signal),
    retry: false,
  });
  const jobs = useQuery({
    queryKey: captioningKeys.jobs,
    queryFn: ({ signal }) => fetchCaptionJobs(signal),
    retry: false,
    refetchInterval: (query) => (query.state.error ? false : 5000),
  });
  const selectedJob = params.get("job") || "";
  const captionViewPath = (name = "") => `${embeddedBasePath}${name ? `/${name}` : ""}${location.search}`;
  function setSelectedJob(jobId: string) {
    setParams((current) => {
      const next = new URLSearchParams();
      for (const key of ["dataset", "subset"]) {
        const value = current.get(key);
        if (value !== null) next.set(key, value);
      }
      if (jobId) next.set("job", jobId);
      return next;
    }, { replace: true });
  }

  function setSourceContext(file: string, index: number) {
    setParams((current) => {
      const next = new URLSearchParams(current);
      if (file) next.set("dataset", file);
      else next.delete("dataset");
      next.set("subset", String(index));
      next.delete("job");
      return next;
    }, { replace: true });
  }
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
          <NavLink to={captionViewPath()} end replace state={location.state}>
            图片与任务
          </NavLink>
          <NavLink to={captionViewPath("providers")} replace state={location.state}>接入预设</NavLink>
          <NavLink to={captionViewPath("prompts")} replace state={location.state}>提示词预设</NavLink>
          <NavLink to={captionViewPath("assets")} replace state={location.state}>本地资源</NavLink>
        </nav>
        {(profiles.isPending || profiles.error) && (
          <QueryFeedback query={profiles} label="接入预设" hasData={Boolean(profiles.data)} />
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
                      if (ids.includes(selectedJob)) setSelectedJob("");
                    }}
                  />
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedJob("");
                      setShowJobs(false);
                    }}
                  >
                    新任务
                  </button>
                </header>
                {(jobs.isPending || jobs.error) && (
                  <QueryFeedback query={jobs} label="打标任务" hasData={Boolean(jobs.data)} />
                )}
                {(jobs.data?.jobs || []).map((job) => (
                  <button
                    className="object-row"
                    type="button"
                    key={job.id}
                    data-selected={selectedJob === job.id}
                    onClick={() => {
                      setSelectedJob(job.id);
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
                    onCreated={(id) => setSelectedJob(id)}
                    onSourceChange={setSourceContext}
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
