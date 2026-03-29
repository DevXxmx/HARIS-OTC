import React from 'react';
import { motion } from 'framer-motion';

export default function AnalysisSidebar({ alert }) {
  // When no alert yet, show idle panel
  if (!alert) {
    return (
      <div className="w-full bg-bg-surface border border-border p-5 flex flex-col h-full">
        <h2 className="text-white-soft font-rajdhani font-bold tracking-[0.2em] text-lg border-b border-border pb-3 mb-5">
          AI ANALYSIS
        </h2>
        <div className="flex-1 flex flex-col items-center justify-center gap-3 opacity-40">
          <div className="w-3 h-3 rounded-full bg-grey-mid animate-pulse" />
          <p className="font-mono text-grey-mid text-xs tracking-widest text-center">
            MONITORING ACTIVE<br />AWAITING DETECTION
          </p>
        </div>
      </div>
    );
  }

  const dateObj = new Date(alert.upload_time + 'Z');
  const y = dateObj.getFullYear();
  const m = String(dateObj.getMonth() + 1).padStart(2, '0');
  const d = String(dateObj.getDate()).padStart(2, '0');
  const h = String(dateObj.getHours()).padStart(2, '0');
  const min = String(dateObj.getMinutes()).padStart(2, '0');
  const s = String(dateObj.getSeconds()).padStart(2, '0');
  
  const formattedDate = `${y}.${m}.${d}`;
  const timePart = `${h}:${min}:${s}`;

  return (
    <div className="w-full bg-bg-surface border border-border p-5 flex flex-col h-full">
      <h2 className="text-white-soft font-rajdhani font-bold tracking-[0.2em] text-lg border-b border-border pb-3 mb-5">
        AI ANALYSIS
      </h2>

      <div className="flex-1 flex flex-col gap-8">

        {/* Detection Event */}
        <div>
          <div className="text-xs text-grey-light font-rajdhani tracking-widest mb-3">
            LAST DETECTION EVENT
          </div>
          <motion.div
            key={alert.id}
            initial={{ opacity: 0, x: 8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4 }}
            className="bg-red-dim border border-red-mid text-red-hot font-mono text-xs px-3 py-1.5 uppercase tracking-wider self-start inline-block"
          >
            HAND IN POCKET
          </motion.div>
        </div>

        {/* Capture Info */}
        <div>
          <div className="text-xs text-grey-light font-rajdhani tracking-widest mb-3">
            CAPTURED
          </div>
          <div className="flex flex-col gap-1">
            <div className="font-mono text-white-soft text-sm">{formattedDate}</div>
            <div className="font-mono text-grey-mid text-sm">{timePart}</div>
          </div>
        </div>

        {/* Screenshot filename */}
        <div>
          <div className="text-xs text-grey-light font-rajdhani tracking-widest mb-2">
            FILE
          </div>
          <div className="font-mono text-grey-light text-xs break-all border-l-2 border-red-dim pl-3">
            {alert.filename}
          </div>
        </div>

        {/* Status bar */}
        <div className="mt-auto border-t border-border pt-5">
          <div className="text-xs text-grey-light font-rajdhani tracking-widest mb-3">
            SYSTEM STATUS
          </div>
          <div className="flex gap-1.5 h-4">
            {[1, 2, 3, 4, 5].map((i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, scaleY: 0 }}
                animate={{ opacity: 1, scaleY: 1 }}
                transition={{ delay: i * 0.08, duration: 0.3 }}
                className={`flex-1 shrink-0 ${i <= 4 ? 'bg-red-hot shadow-[0_0_8px_var(--color-red-glow)]' : 'bg-bg-raised border border-border'}`}
              />
            ))}
          </div>
          <div className="text-xs text-grey-mid font-rajdhani tracking-widest mt-2">
            ALERT RECORDED
          </div>
        </div>
      </div>
    </div>
  );
}
