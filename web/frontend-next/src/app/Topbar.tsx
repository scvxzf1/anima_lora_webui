import { NavLink } from 'react-router-dom';

const LINKS = [
  ['/datasets', '数据集蓝图'],
  ['/training', '训练配置'],
  ['/queue', '训练队列'],
  ['/monitor', '当前监控'],
  ['/history', '历史任务'],
] as const;

export function Topbar() {
  return (
    <header className="topbar">
      <strong>Dragon trainer</strong>
      <nav aria-label="主导航">
        {LINKS.map(([to, label]) => (
          <NavLink key={to} to={to} end={to === '/datasets'}>
            {label}
          </NavLink>
        ))}
      </nav>
    </header>
  );
}
