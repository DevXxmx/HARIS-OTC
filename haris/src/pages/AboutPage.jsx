import React from 'react';
import { motion } from 'framer-motion';

export default function AboutPage() {
  return (
    <div className="w-full h-full min-h-[calc(100vh-8rem)] flex items-center justify-center p-6 lg:p-12">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-4xl bg-bg-raised border border-border p-8 lg:p-12 shadow-[0_4px_30px_rgba(0,0,0,0.5)] flex flex-col gap-8 relative overflow-hidden"
      >
        {/* Decorative corner accents */}
        <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-red-hot opacity-50"></div>
        <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-red-hot opacity-50"></div>

        <div className="border-b border-border pb-6 flex items-end justify-between">
          <div>
            <h1 className="text-4xl md:text-5xl font-rajdhani font-bold tracking-widest text-white-soft">
              ABOUT <span className="text-red-hot [text-shadow:0_0_12px_var(--color-red-glow)]">HARIS OTC</span>
            </h1>
            <p className="text-grey-light font-mono text-sm tracking-widest mt-2 uppercase">
              // Threat Recognition & Surveillance Protocol
            </p>
          </div>
          <motion.div 
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
            className="hidden md:block w-16 h-1 bg-red-hot shadow-[0_0_8px_var(--color-red-glow)]"
          />
        </div>

        <div className="space-y-6 text-grey-light font-rajdhani text-lg leading-relaxed">
          <p>
            HARIS (High-Resolution Artificial Recognition & Intelligence System) Over-The-Counter is an advanced surveillance and anomaly detection framework. Engineered for high-stakes environments, HARIS seamlessly integrates edge computing with real-time computer vision to establish a comprehensive security perimeter.
          </p>
          <p>
            By leveraging state-of-the-art machine learning models, HARIS continuously analyzes live video feeds to identify objects, classify potential threats, and generate actionable alerts with sub-second latency. The system is designed for both autonomous operation and seamless human-in-the-loop oversight.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4">
            <div className="bg-bg-surface border border-border p-4 hover:border-red-mid transition-colors group">
              <div className="w-8 h-8 mb-3 border border-red-hot/30 group-hover:bg-red-hot group-hover:border-red-hot transition-all flex items-center justify-center font-mono text-xs font-bold text-white-soft">01</div>
              <h3 className="font-bold text-white-soft tracking-widest mb-2">DETECTION</h3>
              <p className="text-sm">Real-time object classification and bounding box generation across multiple simultaneous streams.</p>
            </div>
            <div className="bg-bg-surface border border-border p-4 hover:border-red-mid transition-colors group">
              <div className="w-8 h-8 mb-3 border border-red-hot/30 group-hover:bg-red-hot group-hover:border-red-hot transition-all flex items-center justify-center font-mono text-xs font-bold text-white-soft">02</div>
              <h3 className="font-bold text-white-soft tracking-widest mb-2">ANALYSIS</h3>
              <p className="text-sm">Continuous threat level assessment with automated risk scoring based on recognized entities.</p>
            </div>
            <div className="bg-bg-surface border border-border p-4 hover:border-red-mid transition-colors group">
              <div className="w-8 h-8 mb-3 border border-red-hot/30 group-hover:bg-red-hot group-hover:border-red-hot transition-all flex items-center justify-center font-mono text-xs font-bold text-white-soft">03</div>
              <h3 className="font-bold text-white-soft tracking-widest mb-2">RESPONSE</h3>
              <p className="text-sm">Instantaneous alert dispatch and logging protocol execution upon threshold breach.</p>
            </div>
          </div>
        </div>
        
        {/* Logos Footer */}
        <div className="mt-4 pt-6 border-t border-border flex flex-col sm:flex-row items-center justify-between gap-6">
          {/* HARIS logo */}
          <div className="flex items-center gap-4">
            <img
              src="/Haris logo final.png"
              alt="HARIS Logo"
              className="h-14 w-auto object-contain"
            />
            <div className="flex flex-col">
              <span className="font-rajdhani font-bold text-white-soft tracking-[0.2em] text-lg">HARIS OTC</span>
              <span className="font-mono text-grey-mid text-xs tracking-widest">SYS.VER: 2.4.1-BETA</span>
            </div>
          </div>

          <div className="h-px w-full sm:w-px sm:h-10 bg-border" />

          {/* One Tech Crew logo */}
          <div className="flex items-center gap-4">
            <img
              src="/with black backround.png"
              alt="One Tech Crew Logo"
              className="h-14 w-auto object-contain"
            />
            <div className="flex flex-col">
              <span className="font-rajdhani font-bold text-white-soft tracking-[0.2em] text-base">ONE TECH CREW</span>
              <span className="font-mono text-grey-mid text-xs tracking-widest">Every dream needs a team</span>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
