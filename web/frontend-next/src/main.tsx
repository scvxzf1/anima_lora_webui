import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import { DragonNextApp } from './app/DragonNextApp';
import './styles/base.css';

const root = document.getElementById('root');

if (!root) {
  throw new Error('Dragon next root element is missing');
}

createRoot(root).render(
  <StrictMode>
    <DragonNextApp />
  </StrictMode>,
);
