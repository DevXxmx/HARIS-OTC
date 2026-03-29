import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

export default function CameraFeed() {
  const [time, setTime] = useState('');

  useEffect(() => {
    const updateTime = () => setTime(new Date().toLocaleTimeString('en-US', { hour12: false }));
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <motion.div
      initial={{ rotateX: 8, scale: 0.96, opacity: 0 }}
      animate={{ rotateX: 0, scale: 1, opacity: 1 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
      className="relative w-full h-full min-h-[400px] lg:min-h-[500px] bg-[#050505] border border-border overflow-hidden"
      style={{ perspective: '1000px' }}
    >
      {/* Live AI Video Stream from Backend */}
      <img 
        src="http://localhost:5000/api/stream"
        alt="Live Camera Feed"
        className="absolute inset-0 w-full h-full object-cover filter grayscale-[20%] contrast-[1.05]"
      />

      {/* CSS Scanline Sweep */}
      <div className="absolute inset-0 pointer-events-none z-10 scanline-sweep" />
      
      {/* Corner Brackets */}
      <div className="absolute top-4 left-4 w-8 h-8 border-t-[3px] border-l-[3px] border-red-dim z-20" />
      <div className="absolute top-4 right-4 w-8 h-8 border-t-[3px] border-r-[3px] border-red-dim z-20" />
      <div className="absolute bottom-4 left-4 w-8 h-8 border-b-[3px] border-l-[3px] border-red-dim z-20" />
      <div className="absolute bottom-4 right-4 w-8 h-8 border-b-[3px] border-r-[3px] border-red-dim z-20" />

      {/* Top Left Badge */}
      <div className="absolute top-6 left-6 flex items-center gap-2 z-20">
        <motion.div 
          animate={{ opacity: [1, 0, 1] }} 
          transition={{ repeat: Infinity, duration: 1.5 }}
          className="w-3 h-3 bg-red-hot rounded-full shadow-[0_0_8px_var(--color-red-glow)]"
        />
        <span className="text-red-hot font-rajdhani font-bold tracking-[0.15em] text-sm">
          REC CAM-01
        </span>
      </div>

      {/* Top Right Badge (Timestamp) */}
      <div className="absolute top-6 right-6 z-20">
        <span className="text-white-soft font-mono text-sm tracking-widest bg-black/50 px-2 py-1">
          {time}
        </span>
      </div>



      {/* Internal styles for custom animations */}
      <style>{`
        .scanline-sweep {
          background: linear-gradient(to bottom, transparent, rgba(255,255,255,0.08), transparent);
          height: 100%;
          animation: sweep 4s linear infinite;
        }
        @keyframes sweep {
          0% { transform: translateY(-100%); }
          100% { transform: translateY(100%); }
        }
      `}</style>
    </motion.div>
  );
}
