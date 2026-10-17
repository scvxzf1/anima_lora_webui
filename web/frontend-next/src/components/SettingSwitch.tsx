import * as Switch from '@radix-ui/react-switch';
import type { ComponentPropsWithoutRef } from 'react';
import './SettingSwitch.css';

export function SettingSwitch({ className, ...props }: ComponentPropsWithoutRef<typeof Switch.Root>) {
  return (
    <Switch.Root {...props} className={['setting-switch', className].filter(Boolean).join(' ')}>
      <Switch.Thumb className="setting-switch-thumb" />
    </Switch.Root>
  );
}
