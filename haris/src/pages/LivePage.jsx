import React, { useState, useEffect, useRef } from 'react';
import CameraFeed from '../components/CameraFeed';
import AnalysisSidebar from '../components/AnalysisSidebar';
import LastAlertStrip from '../components/LastAlertStrip';
import { useLatestAlert } from '../hooks/useLatestAlert';

export default function LivePage() {
  const lastAlert = useLatestAlert(8000);

  return (
    <div className="w-full flex flex-col h-full min-h-[calc(100vh-8rem)]">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1">
        {/* Camera Feed takes 2/3 */}
        <div className="lg:col-span-2 h-full flex flex-col">
          <CameraFeed />
        </div>

        {/* Sidebar takes 1/3 */}
        <div className="lg:col-span-1 h-full flex flex-col">
          <AnalysisSidebar alert={lastAlert} />
        </div>
      </div>

      {/* Last Alert Strip below the grid */}
      <div className="mt-6 shrink-0">
        <LastAlertStrip alert={lastAlert} />
      </div>
    </div>
  );
}
