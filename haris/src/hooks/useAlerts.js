import { useState, useEffect } from 'react';
import { API_BASE } from '../api';

export function useAlerts(pollingIntervalMs = 10000) {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    let isMounted = true;
    let timeoutId = null;

    const fetchAlerts = async () => {
      const controller = new AbortController();
      try {
        const response = await fetch(`${API_BASE}/api/alerts`, { signal: controller.signal });
        if (!response.ok) throw new Error('API failed');
        
        const data = await response.json();
        
        if (isMounted) {
          setAlerts(data);
          setError(false);
          setLoading(false);
        }
      } catch (err) {
        if (err.name !== 'AbortError' && isMounted) {
          setError(true);
          setLoading(false);
        }
      } finally {
        if (isMounted && pollingIntervalMs) {
          timeoutId = setTimeout(fetchAlerts, pollingIntervalMs);
        }
      }
    };

    fetchAlerts();

    return () => {
      isMounted = false;
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [pollingIntervalMs]);

  return { alerts, loading, error };
}
