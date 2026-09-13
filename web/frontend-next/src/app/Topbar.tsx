import { NavLink } from "react-router-dom";
import {
  Box,
  Captions,
  Cpu,
  Database,
  History,
  ListTodo,
  Monitor,
  Settings,
  Layers3,
  Sun,
  Moon,
  Menu,
  ExternalLink,
} from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "../components/ui/Button";

const LINKS = [
  ["/training", "训练配置", Box],
  ["/datasets", "数据集蓝图", Database],
  ["/queue", "训练队列", ListTodo],
  ["/monitor", "当前监控", Monitor],
  ["/history", "历史任务", History],
  ["/models", "模型配置", Cpu],
  ["/captioning", "打标工作台", Captions],
  ["/settings", "全局设置", Settings],
] as const;

export function Topbar() {
  const [expanded, setExpanded] = useState(false);
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem("dragon-next-ui-v1-theme") === "light"
        ? "light"
        : "dark";
    } catch {
      return "dark";
    }
  });
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("dragon-next-ui-v1-theme", theme);
    } catch {
      /* Private browsing can deny storage. */
    }
  }, [theme]);
  return (
    <aside className="app-navigation" data-expanded={expanded}>
      <div className="app-brand">
        <Layers3 size={23} />
        <strong>
          Dragon<span>TRAINER WORKSPACE</span>
        </strong>
        <Button
          variant="ghost"
          size="icon"
          className="nav-toggle"
          aria-label="切换导航"
          aria-expanded={expanded}
          onClick={() => setExpanded(!expanded)}
        >
          <Menu size={19} />
        </Button>
      </div>
      <p className="nav-section-label">工作空间</p>
      <nav aria-label="主导航" className="app-links">
        {LINKS.map(([to, label, Icon]) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/datasets"}
            onClick={() => setExpanded(false)}
          >
            <Icon size={18} aria-hidden="true" />
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="nav-footer">
        <details>
          <summary>辅助工具</summary>
          <a href="/?ui=dragon#image-test">
            生图测试 <ExternalLink size={13} />
          </a>
          <a href="/?ui=dragon#weight-analysis">
            权重分析 <ExternalLink size={13} />
          </a>
          <a href="/?ui=dragon#environment">
            环境检测 <ExternalLink size={13} />
          </a>
        </details>
        <Button
          variant="ghost"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        >
          {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
          {theme === "dark" ? "浅色外观" : "深色外观"}
        </Button>
        <a href="/?ui=dragon">
          旧版界面 <ExternalLink size={13} />
        </a>
      </div>
    </aside>
  );
}
