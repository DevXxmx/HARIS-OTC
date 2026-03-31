import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, AlertTriangle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

import { API_BASE } from '../api';

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isFocused, setIsFocused] = useState(null);
  const [error, setError] = useState('');
  const [shake, setShake] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });

      if (response.ok) {
        localStorage.setItem('isAuthenticated', 'true');
        navigate('/live');
      } else {
        setLoading(false);
        setError('ACCESS DENIED — Invalid credentials.');
        setShake(true);
        setTimeout(() => setShake(false), 600);
        setPassword('');
      }
    } catch (err) {
      setLoading(false);
      setError('SYSTEM OFFLINE — Cannot verify credentials.');
      setShake(true);
      setTimeout(() => setShake(false), 600);
    }
  };

  return (
    <div className="w-full h-full min-h-[calc(100vh-8rem)] flex items-center justify-center p-6">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={shake
          ? { opacity: 1, scale: 1, x: [0, -12, 12, -8, 8, -4, 4, 0] }
          : { opacity: 1, scale: 1, x: 0 }
        }
        transition={{ duration: shake ? 0.5 : 0.4 }}
        className="w-full max-w-md bg-bg-raised border border-border p-8 shadow-[0_0_40px_rgba(0,0,0,0.8)] relative"
      >
        {/* Scanner animation line */}
        <motion.div
          animate={{ top: ['0%', '100%', '0%'] }}
          transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
          className="absolute left-0 right-0 h-[1px] bg-red-hot/20 z-0 pointer-events-none"
        />

        {/* Header */}
        <div className="relative z-10 flex flex-col items-center mb-8">
          <div className="w-16 h-16 border border-red-hot/50 flex items-center justify-center mb-4 relative before:absolute before:inset-0 before:bg-red-hot/10 before:animate-pulse">
            <Search className="text-red-hot w-8 h-8" />
          </div>
          <h1 className="text-3xl font-rajdhani font-bold tracking-[0.2em] text-white-soft text-center">
            SYS.<span className="text-red-hot">AUTH</span>
          </h1>
          <p className="text-grey-light font-mono text-xs tracking-widest mt-2 uppercase">
            // Authorized Personnel Only
          </p>
        </div>

        {/* Error Banner */}
        <AnimatePresence>
          {error && (
            <motion.div
              key="error"
              initial={{ opacity: 0, y: -8, height: 0 }}
              animate={{ opacity: 1, y: 0, height: 'auto' }}
              exit={{ opacity: 0, y: -8, height: 0 }}
              transition={{ duration: 0.25 }}
              className="mb-5 flex items-center gap-2 bg-red-dim border border-red-hot/60 px-4 py-2.5 text-red-hot font-mono text-xs tracking-widest"
            >
              <AlertTriangle className="w-4 h-4 shrink-0" />
              {error}
            </motion.div>
          )}
        </AnimatePresence>

        <form onSubmit={handleSubmit} className="relative z-10 space-y-6">

          {/* Operator ID field */}
          <div className="space-y-1 group">
            <label className="text-xs font-mono text-grey-light tracking-widest uppercase block flex justify-between">
              <span>Operator ID</span>
              <span className={`transition-colors ${isFocused === 'username' ? 'text-red-hot' : 'text-transparent'}`}>[ ACTIVE ]</span>
            </label>
            <div className={`relative border bg-bg-surface transition-colors ${error ? 'border-red-hot/60' : isFocused === 'username' ? 'border-red-hot/50 shadow-[0_0_10px_var(--color-red-glow)]' : 'border-border hover:border-grey-mid'
              }`}>
              <div className={`absolute left-0 top-0 bottom-0 w-1 transition-colors ${isFocused === 'username' ? 'bg-red-hot' : 'bg-transparent'}`} />
              <input
                type="text"
                value={username}
                onChange={(e) => { setUsername(e.target.value); setError(''); }}
                onFocus={() => setIsFocused('username')}
                onBlur={() => setIsFocused(null)}
                className="w-full bg-transparent border-none text-white-soft font-mono text-sm px-4 py-3 placeholder:text-grey-mid focus:outline-none"
                placeholder="ENTER ID..."
                required
                autoComplete="username"
              />
            </div>
          </div>

          {/* Passcode field */}
          <div className="space-y-1 group">
            <label className="text-xs font-mono text-grey-light tracking-widest uppercase block flex justify-between">
              <span>Passcode</span>
              <span className={`transition-colors ${isFocused === 'password' ? 'text-red-hot' : 'text-transparent'}`}>[ ACTIVE ]</span>
            </label>
            <div className={`relative border bg-bg-surface transition-colors ${error ? 'border-red-hot/60' : isFocused === 'password' ? 'border-red-hot/50 shadow-[0_0_10px_var(--color-red-glow)]' : 'border-border hover:border-grey-mid'
              }`}>
              <div className={`absolute left-0 top-0 bottom-0 w-1 transition-colors ${isFocused === 'password' ? 'bg-red-hot' : 'bg-transparent'}`} />
              <input
                type="password"
                value={password}
                onChange={(e) => { setPassword(e.target.value); setError(''); }}
                onFocus={() => setIsFocused('password')}
                onBlur={() => setIsFocused(null)}
                className="w-full bg-transparent border-none text-white-soft font-mono text-sm px-4 py-3 placeholder:text-grey-mid focus:outline-none"
                placeholder="••••••••"
                required
                autoComplete="current-password"
              />
            </div>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading}
            className="w-full mt-4 border border-red-hot bg-red-dim hover:bg-red-mid text-white-soft font-rajdhani font-bold tracking-[0.2em] py-3 transition-colors relative overflow-hidden group disabled:opacity-60 disabled:cursor-not-allowed"
          >
            <span className="relative z-10">
              {loading ? 'AUTHENTICATING...' : 'INITIALIZE UPLINK'}
            </span>
            {!loading && (
              <div className="absolute inset-0 bg-red-hot/20 translate-x-[-100%] group-hover:translate-x-0 transition-transform duration-300 ease-out" />
            )}
            {loading && (
              <motion.div
                className="absolute bottom-0 left-0 h-[2px] bg-red-hot"
                initial={{ width: '0%' }}
                animate={{ width: '100%' }}
                transition={{ duration: 0.6, ease: 'easeInOut' }}
              />
            )}
          </button>
        </form>

        <div className="mt-8 pt-4 border-t border-border flex justify-between text-[10px] font-mono text-grey-mid tracking-widest text-center">
          <span>SEC.LEVEL: ALPHA</span>
          <span>NODE: 0x4B2</span>
        </div>
      </motion.div>
    </div>
  );
}
