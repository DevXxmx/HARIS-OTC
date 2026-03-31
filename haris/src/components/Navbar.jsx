import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import { motion } from 'framer-motion';
import { API_BASE } from '../api';

export default function Navbar() {
  const logoText = "HARIS";
  const [systemOnline, setSystemOnline] = useState(false);
  const [hasActiveAlert, setHasActiveAlert] = useState(false);

  useEffect(() => {
    const check = () => {
      fetch(`${API_BASE}/api/alerts/latest`)
        .then(res => res.json())
        .then(data => {
          setSystemOnline(true);
          if (data) {
            // Append 'Z' so the browser treats it as UTC, not local time
            const alertTime = new Date(data.upload_time + 'Z');
            const now = new Date();
            const secondsAgo = (now - alertTime) / 1000;
            setHasActiveAlert(secondsAgo < 30);
          } else {
            setHasActiveAlert(false);
          }
        })
        .catch(() => {
          setSystemOnline(false);
          setHasActiveAlert(false);
        });
    };
    check();
    const interval = setInterval(check, 8000);
    return () => clearInterval(interval);
  }, []);

  // Status chip config
  const statusConfig = systemOnline === false
    ? { dotClass: 'bg-grey-mid', pingClass: '', label: 'BACKEND OFFLINE' }
    : hasActiveAlert
      ? { dotClass: 'bg-red-hot shadow-[0_0_8px_var(--color-red-glow)]', pingClass: 'bg-red-hot', label: 'ALERT ACTIVE' }
      : { dotClass: 'bg-green-500', pingClass: 'bg-green-500', label: 'SYSTEM ONLINE' };

  return (
    <nav className="w-full bg-bg-surface border-b border-border flex items-center justify-between px-6 h-16 shrink-0 relative z-20">
      {/* Left side — HARIS Logo */}
      <div className="flex items-center gap-3">
        <img
          src="/Haris logo final.png"
          alt="HARIS"
          className="h-9 w-auto object-contain"
        />
      </div>

      {/* Center Links */}
      <div className="flex items-center gap-8 h-full">
        <NavLink
          to="/live"
          className={({ isActive }) =>
            `relative h-full flex items-center transition-colors hover:text-grey-light text-sm font-semibold tracking-widest ${isActive ? 'text-white-soft [text-shadow:0_0_8px_var(--color-red-glow)]' : 'text-grey-mid'}`
          }
        >
          {({ isActive }) => (
            <>
              LIVE FEED
              {isActive && (
                <motion.div
                  layoutId="nav-underline"
                  className="absolute bottom-0 left-0 right-0 h-[2px] bg-red-hot shadow-[0_0_12px_var(--color-red-glow)]"
                />
              )}
            </>
          )}
        </NavLink>
        <NavLink
          to="/alerts"
          className={({ isActive }) =>
            `relative h-full flex items-center transition-colors hover:text-grey-light text-sm font-semibold tracking-widest ${isActive ? 'text-white-soft [text-shadow:0_0_8px_var(--color-red-glow)]' : 'text-grey-mid'}`
          }
        >
          {({ isActive }) => (
            <>
              ALERTS
              {isActive && (
                <motion.div
                  layoutId="nav-underline"
                  className="absolute bottom-0 left-0 right-0 h-[2px] bg-red-hot shadow-[0_0_12px_var(--color-red-glow)]"
                />
              )}
            </>
          )}
        </NavLink>
        <NavLink
          to="/about"
          className={({ isActive }) =>
            `relative h-full flex items-center transition-colors hover:text-grey-light text-sm font-semibold tracking-widest ${isActive ? 'text-white-soft [text-shadow:0_0_8px_var(--color-red-glow)]' : 'text-grey-mid'}`
          }
        >
          {({ isActive }) => (
            <>
              ABOUT
              {isActive && (
                <motion.div
                  layoutId="nav-underline"
                  className="absolute bottom-0 left-0 right-0 h-[2px] bg-red-hot shadow-[0_0_12px_var(--color-red-glow)]"
                />
              )}
            </>
          )}
        </NavLink>
      </div>

      {/* Right corner — live status chip & logout */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <div className="relative flex h-3 w-3 items-center justify-center">
            {statusConfig.pingClass && (
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${statusConfig.pingClass} opacity-75`} />
            )}
            <span className={`relative inline-flex rounded-full h-2 w-2 ${statusConfig.dotClass}`} />
          </div>
          <span className="text-xs text-grey-light font-mono tracking-widest mt-0.5">
            {statusConfig.label}
          </span>
        </div>
        
        <button 
          onClick={() => {
            localStorage.removeItem('isAuthenticated');
            window.location.href = '/login';
          }}
          className="text-xs font-mono tracking-widest text-grey-mid hover:text-red-hot transition-colors border border-transparent hover:border-red-hot border-b px-2 py-1"
        >
          LOGOUT
        </button>
      </div>
    </nav>
  );
}
