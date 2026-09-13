"""Frontend task dispatch must not launch the training service."""

from scripts.tasks import web


def test_frontend_build_uses_project_directory(monkeypatch):
    calls = []
    monkeypatch.setattr(web.shutil, "which", lambda name: "/bin/pnpm")
    monkeypatch.setattr(web, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    web.cmd_frontend_build([])
    assert calls[0][0] == ["/bin/pnpm", "--dir", str(web.ROOT / "web" / "frontend-next"), "build"]
    assert calls[0][1]["cwd"] == str(web.ROOT)


def test_frontend_check_runs_types_before_tests(monkeypatch):
    calls = []
    monkeypatch.setattr(web, "_frontend_command", lambda command, extra: calls.append((command, extra)))
    web.cmd_frontend_check(["--reporter=dot"])
    assert calls == [("typecheck", []), ("test", ["--reporter=dot"])]
