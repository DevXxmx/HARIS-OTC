import React, { useState, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CameraOff, ShieldAlert, Search, Calendar } from 'lucide-react';
import AlertCard from '../components/AlertCard';
import AnalyticsChart from '../components/AnalyticsChart';
import { useAlerts } from '../hooks/useAlerts';

export default function AlertsPage() {
  const { alerts, loading, error } = useAlerts(10000);
  const [selectedAlertId, setSelectedAlertId] = useState(null);

  const [searchQuery, setSearchQuery] = useState('');
  const [dateFilter, setDateFilter] = useState('ALL'); // 'ALL' or 'TODAY'

  const filteredAlerts = useMemo(() => {
    return alerts.filter(alert => {
      // Calculate local datetime once per alert
      const alertDate = new Date(alert.upload_time + 'Z');
      const h = String(alertDate.getHours()).padStart(2, '0');
      const min = String(alertDate.getMinutes()).padStart(2, '0');
      const s = String(alertDate.getSeconds()).padStart(2, '0');
      const localTime = `${h}:${min}:${s}`;

      const matchesSearch = searchQuery === '' || 
        alert.id.toString().includes(searchQuery.replace('#', '')) ||
        alert.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
        localTime.includes(searchQuery);

      let matchesDate = true;
      if (dateFilter === 'TODAY') {
        const today = new Date();
        matchesDate = 
          alertDate.getDate() === today.getDate() &&
          alertDate.getMonth() === today.getMonth() &&
          alertDate.getFullYear() === today.getFullYear();
      }
      return matchesSearch && matchesDate;
    });
  }, [alerts, searchQuery, dateFilter]);

  const handleToggle = useCallback((id) => {
    setSelectedAlertId(prevId => (prevId === id ? null : id));
  }, []);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.08 }
    }
  };

  return (
    <div className="w-full pb-10 min-h-screen">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between border-b border-border pb-6 mb-8 gap-4">
        <div>
          <h1 className="text-3xl font-rajdhani font-bold tracking-[0.2em] text-white-soft flex items-center gap-4">
            ALERT LOG
            <span className="bg-red-dim text-red-hot text-sm px-3 py-1 font-mono tracking-widest border border-red-hot shadow-[0_0_8px_var(--color-red-glow)]">
              {loading ? '···' : `${filteredAlerts.length} / ${alerts.length} EVENTS`}
            </span>
          </h1>
        </div>
      </div>

      {/* Filters Bar */}
      <div className="flex flex-col sm:flex-row gap-4 mb-8">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-grey-mid" />
          <input
            type="text"
            placeholder="Search by ID, time (14:30), or filename..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-bg-raised border border-border text-white-soft font-mono text-sm py-2.5 pl-10 pr-4 focus:outline-none focus:border-red-mid transition-colors placeholder:text-grey-mid"
          />
        </div>
        <div className="flex gap-2 shrink-0">
          <button
            onClick={() => setDateFilter('ALL')}
            className={`px-4 py-2.5 font-rajdhani tracking-widest text-sm font-bold border transition-colors ${
              dateFilter === 'ALL'
                ? 'bg-red-dim border-red-hot text-red-hot shadow-[0_0_8px_var(--color-red-glow)]'
                : 'bg-bg-raised border-border text-grey-mid hover:text-white-soft hover:border-grey-mid'
            }`}
          >
            ALL TIME
          </button>
          <button
            onClick={() => setDateFilter('TODAY')}
            className={`px-4 py-2.5 font-rajdhani tracking-widest text-sm font-bold border transition-colors flex items-center gap-2 ${
              dateFilter === 'TODAY'
                ? 'bg-red-dim border-red-hot text-red-hot shadow-[0_0_8px_var(--color-red-glow)]'
                : 'bg-bg-raised border-border text-grey-mid hover:text-white-soft hover:border-grey-mid'
            }`}
          >
            <Calendar className="w-4 h-4" /> TODAY
          </button>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full bg-red-dim border border-red-mid text-red-hot font-mono text-sm px-4 py-3 mb-6 tracking-wide"
        >
          ⚠&nbsp;&nbsp;CANNOT REACH BACKEND — Make sure Flask is running (python app.py)
        </motion.div>
      )}

      {/* Loading Skeletons */}
      {loading && (
        <div className="flex flex-col gap-3">
          {[0, 1, 2, 3].map(i => (
            <div
              key={i}
              className="bg-bg-raised border border-border h-[64px] animate-pulse"
              style={{ animationDelay: `${i * 150}ms` }}
            />
          ))}
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && alerts.length === 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex flex-col items-center justify-center py-24 opacity-60"
        >
          <CameraOff className="w-12 h-12 text-grey-mid mb-5 opacity-50" />
          <h3 className="font-rajdhani text-2xl tracking-[0.2em] text-grey-mid mb-2">
            NO DETECTIONS RECORDED
          </h3>
          <p className="text-grey-mid text-sm font-mono text-center max-w-sm">
            System is monitoring. Alerts will appear here automatically when behavior is detected.
          </p>
        </motion.div>
      )}

      {/* No Filter Results State */}
      {!loading && !error && alerts.length > 0 && filteredAlerts.length === 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex flex-col items-center justify-center py-16 opacity-60"
        >
          <Search className="w-10 h-10 text-grey-mid mb-4 opacity-50" />
          <h3 className="font-rajdhani text-xl tracking-[0.2em] text-grey-mid mb-2">
            NO MATCHES FOUND
          </h3>
          <p className="text-grey-mid text-sm font-mono text-center">
            Try adjusting your search criteria or date filter.
          </p>
        </motion.div>
      )}

      {/* Analytics Chart */}
      {!loading && !error && filteredAlerts.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <AnalyticsChart alerts={filteredAlerts} />
        </motion.div>
      )}

      {/* Alert List */}
      {!loading && filteredAlerts.length > 0 && (
        <AnimatePresence mode="wait">
          <motion.div
            layout
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="flex flex-col gap-5 mt-6 mb-8"
          >
            {filteredAlerts.map(alert => (
              <AlertCard
                key={alert.id}
                alert={alert}
                isSelected={selectedAlertId === alert.id}
                onToggle={handleToggle}
              />
            ))}
          </motion.div>
        </AnimatePresence>
      )}
    </div>
  );
}
