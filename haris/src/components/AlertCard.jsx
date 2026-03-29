import React from 'react';
import { ChevronDown } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { API_BASE } from '../api';

function formatTimestamp(isoString) {
  // Append 'Z' to treat python UTC as actual UTC.
  const dateObj = new Date(isoString + 'Z');
  const y = dateObj.getFullYear();
  const m = String(dateObj.getMonth() + 1).padStart(2, '0');
  const d = String(dateObj.getDate()).padStart(2, '0');
  const h = String(dateObj.getHours()).padStart(2, '0');
  const min = String(dateObj.getMinutes()).padStart(2, '0');
  const s = String(dateObj.getSeconds()).padStart(2, '0');

  return {
    date: `${y}.${m}.${d}`,
    time: `${h}:${min}:${s}`,
  };
}

function formatDisplayTimestamp(isoString) {
  const dateObj = new Date(isoString + 'Z');
  const y = dateObj.getFullYear();
  const m = String(dateObj.getMonth() + 1).padStart(2, '0');
  const d = String(dateObj.getDate()).padStart(2, '0');
  const h = String(dateObj.getHours()).padStart(2, '0');
  const min = String(dateObj.getMinutes()).padStart(2, '0');
  const s = String(dateObj.getSeconds()).padStart(2, '0');

  return `${y} · ${m} · ${d}   —   ${h} : ${min} : ${s}`;
}

const AlertCard = React.memo(({ alert, isSelected, onToggle }) => {
  const handleToggle = React.useCallback(() => onToggle(alert.id), [onToggle, alert.id]);
  const { date, time } = formatTimestamp(alert.upload_time);

  return (
    <motion.div
      layout
      variants={{
        hidden: { opacity: 0, x: -20 },
        visible: { opacity: 1, x: 0 },
      }}
      onClick={handleToggle}
      className={`bg-bg-raised border cursor-pointer transition-colors duration-300 relative group overflow-hidden shrink-0 ${
        isSelected
          ? 'border-red-hot bg-red-dim'
          : 'border-border hover:border-red-mid hover:opacity-100 opacity-80'
      }`}
    >
      {/* Left accent bar when selected */}
      <AnimatePresence>
        {isSelected && (
          <motion.div
            layoutId={`accent-${alert.id}`}
            className="absolute left-0 top-0 bottom-0 w-1 bg-red-hot z-20"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          />
        )}
      </AnimatePresence>

      {/* Hover red glow overlay */}
      <div className="absolute inset-0 opacity-0 group-hover:opacity-[0.03] transition-opacity duration-300 pointer-events-none bg-gradient-to-r from-[var(--color-red-glow)] to-transparent" />

      {/* CARD HEADER */}
      <motion.div layout className="p-5 flex items-center justify-between relative z-10 w-full gap-6">

        {/* Left — ID */}
        <div className="font-rajdhani font-bold text-grey-mid text-lg leading-none shrink-0 pl-1 min-w-[3rem]">
          #{alert.id.toString().padStart(3, '0')}
        </div>

        {/* Center — Label + date/time */}
        <div className="flex flex-col min-w-0 flex-1">
          <div className="font-mono text-grey-light text-sm tracking-widest">
            MOTION DETECTED
          </div>
          <div className="flex gap-3 mt-1">
            <span className="font-mono text-grey-mid text-xs">{date}</span>
            <span className="font-mono text-grey-mid text-xs">{time}</span>
          </div>
        </div>

        {/* Right — Chevron */}
        <motion.div
          animate={{ rotate: isSelected ? 180 : 0 }}
          transition={{ duration: 0.25 }}
          className="text-grey-mid mr-2 shrink-0"
        >
          <ChevronDown className="w-5 h-5" />
        </motion.div>
      </motion.div>

      {/* EXPANDED DETAIL PANEL */}
      <AnimatePresence mode="wait">
        {isSelected && (
          <motion.div
            layout
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.35, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div className="border-t border-border mx-4 p-4 flex flex-col gap-4 relative z-10">

              {/* Block 1 — Screenshot */}
              <motion.div
                initial={{ y: 10, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.1, duration: 0.4 }}
                className="relative aspect-video w-full bg-bg-deep border border-border overflow-hidden"
              >
                <img
                  src={`${API_BASE}/static/screenshots/${alert.filename}`}
                  alt="Detection screenshot"
                  className="w-full h-full object-contain"
                  onError={(e) => {
                    e.target.style.display = 'none';
                    e.target.nextSibling.style.display = 'flex';
                  }}
                />
                {/* Fallback if image fails */}
                <div
                  style={{ display: 'none' }}
                  className="w-full h-full items-center justify-center text-grey-mid font-mono text-sm"
                >
                  SCREENSHOT UNAVAILABLE
                </div>

                {/* Scanline overlay */}
                <div className="absolute inset-0 pointer-events-none opacity-20 mix-blend-overlay bg-[repeating-linear-gradient(0deg,transparent,transparent_1px,rgba(255,255,255,0.1)_1px,rgba(255,255,255,0.1)_2px)]" />

                {/* Top-left badge */}
                <div className="absolute top-2 left-2 bg-black/60 text-grey-light font-mono text-xs px-2 py-1 border border-border">
                  CAM · 01
                </div>

                {/* Bottom-right time badge */}
                <div className="absolute bottom-2 right-2 bg-black/60 text-red-hot font-mono text-xs px-2 py-1 border border-red-dim">
                  {time}
                </div>
              </motion.div>

              {/* Block 2 — Captured At */}
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2, duration: 0.3 }}
              >
                <div className="text-xs text-grey-mid uppercase tracking-widest font-rajdhani mb-1">
                  CAPTURED AT
                </div>
                <div className="font-mono text-white-soft text-base tracking-wider">
                  {formatDisplayTimestamp(alert.upload_time)}
                </div>
              </motion.div>

            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
});

export default AlertCard;
