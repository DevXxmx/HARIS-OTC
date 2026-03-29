import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function LastAlertStrip({ alert }) {
  const dateObj = alert ? new Date(alert.upload_time + 'Z') : null;
  const h = dateObj ? String(dateObj.getHours()).padStart(2, '0') : '';
  const min = dateObj ? String(dateObj.getMinutes()).padStart(2, '0') : '';
  const s = dateObj ? String(dateObj.getSeconds()).padStart(2, '0') : '';
  const localTime = `${h}:${min}:${s}`;
  return (
    <div className="w-full mt-4">
      <h3 className="text-white-soft font-rajdhani tracking-widest text-sm mb-2">
        LAST DETECTED ALERT
      </h3>
      <AnimatePresence mode="wait">
        {!alert ? (
          <motion.div
            key="no-alert"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="bg-bg-raised border border-border p-3 font-mono text-sm text-grey-mid text-center tracking-widest"
          >
            — SYSTEM MONITORING — NO ALERTS RECORDED —
          </motion.div>
        ) : (
          <motion.div
            key={alert.id}
            initial={{ opacity: 0, x: -10 }}
            animate={{
              opacity: [1, 0.4, 1],
              x: 0,
              boxShadow: ['0 0 0px var(--color-red-glow)', '0 0 20px var(--color-red-glow)', '0 0 0px var(--color-red-glow)'],
              borderColor: ['var(--color-red-mid)', 'var(--color-red-hot)', 'var(--color-red-mid)']
            }}
            transition={{ duration: 0.8, ease: "easeInOut" }}
            className="bg-red-dim border-l-4 p-3 font-mono text-sm text-white-soft flex items-center"
          >
            <span>
              [{localTime}]&nbsp;&nbsp;MOTION DETECTED&nbsp;&nbsp;—&nbsp;&nbsp;{alert.filename}
            </span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
