import React, { useMemo } from 'react';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';

export default function AnalyticsChart({ alerts }) {
  const chartData = useMemo(() => {
    if (!alerts || alerts.length === 0) return [];

    // Group by hour
    const hourCounts = {};

    alerts.forEach(alert => {
      const dateObj = new Date(alert.upload_time + 'Z');
      const h = String(dateObj.getHours()).padStart(2, '0');
      const hourKey = `${h}:00`;
      hourCounts[hourKey] = (hourCounts[hourKey] || 0) + 1;
    });

    // We want to sort the hours and map them to an array
    const sortedHours = Object.keys(hourCounts).sort();
    
    return sortedHours.map(hour => ({
      time: hour,
      events: hourCounts[hour]
    }));
  }, [alerts]);

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-bg-raised border border-border p-3">
          <p className="font-rajdhani text-white-soft text-lg font-bold flex items-center gap-2">
             <span className="w-2 h-2 bg-red-hot outline outline-1 outline-offset-1 outline-red-hot"></span>
             {label}
          </p>
          <p className="font-mono text-grey-mid text-sm mt-1">
            {payload[0].value} Detections
          </p>
        </div>
      );
    }
    return null;
  };

  if (chartData.length === 0) return null;

  return (
    <div className="w-full bg-bg-surface border border-border p-5 mb-8">
      <h3 className="text-white-soft font-rajdhani tracking-[0.2em] font-bold mb-6 flex items-center justify-between">
        ACTIVITY TIMELINE
        <span className="text-grey-mid font-mono text-xs tracking-normal">HOURLY DETECTIONS</span>
      </h3>
      <div className="w-full h-[200px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="colorEvents" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--color-red-hot)" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="var(--color-red-hot)" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#222" vertical={false} />
            <XAxis 
              dataKey="time" 
              stroke="#666" 
              tick={{ fill: '#666', fontSize: 12, fontFamily: 'monospace' }} 
              axisLine={false}
              tickLine={false}
              dy={10}
            />
            <YAxis 
              stroke="#666" 
              tick={{ fill: '#666', fontSize: 12, fontFamily: 'monospace' }} 
              axisLine={false}
              tickLine={false}
              dx={-10}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#444', strokeWidth: 1, strokeDasharray: '4 4' }} />
            <Area 
              type="monotone" 
              dataKey="events" 
              stroke="var(--color-red-hot)" 
              strokeWidth={2}
              fillOpacity={1} 
              fill="url(#colorEvents)" 
              activeDot={{ r: 6, fill: "var(--color-red-hot)", stroke: "var(--color-red-glow)", strokeWidth: 2 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
