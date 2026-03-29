import { useState, useEffect, useRef } from 'react';
import { API_BASE } from '../api';

export function useLatestAlert(pollingIntervalMs = 8000) {
  const [latestAlert, setLatestAlert] = useState(null);
  const lastAlertIdRef = useRef(null);

  useEffect(() => {
    let isMounted = true;
    let timeoutId = null;

    const fetchLatest = async () => {
      const controller = new AbortController();
      try {
        const response = await fetch(`${API_BASE}/api/alerts/latest`, { signal: controller.signal });
        if (!response.ok) throw new Error('API failed');

        const data = await response.json();
        
        if (isMounted && data && data.id !== lastAlertIdRef.current) {
          lastAlertIdRef.current = data.id;
          setLatestAlert(data);
        }
      } catch (err) {
        // Silently ignore polling errors for the sidebar strip
      } finally {
        if (isMounted && pollingIntervalMs) {
          timeoutId = setTimeout(fetchLatest, pollingIntervalMs);
        }
      }
    };

    fetchLatest();

    return () => {
      isMounted = false;
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [pollingIntervalMs]);

  return latestAlert;
}
