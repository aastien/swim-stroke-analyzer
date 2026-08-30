// API Configuration
// Development always targets the local Flask backend. Remote URLs (including
// the production Render backend) are ignored when NODE_ENV !== 'production'.
const DEFAULT_LOCAL_API = 'http://127.0.0.1:5001/api';

function isLocalApiUrl(value) {
  try {
    const url = new URL(value);
    return url.hostname === '127.0.0.1' || url.hostname === 'localhost';
  } catch {
    return false;
  }
}

function resolveApiBaseUrl() {
  const configured = process.env.REACT_APP_API_URL;
  const isDev = process.env.NODE_ENV !== 'production';

  if (isDev) {
    if (configured && isLocalApiUrl(configured)) {
      return configured.replace(/\/$/, '');
    }
    if (configured && !isLocalApiUrl(configured)) {
      console.warn(
        `Ignoring non-local REACT_APP_API_URL (${configured}) in development; using ${DEFAULT_LOCAL_API}`
      );
    }
    return DEFAULT_LOCAL_API;
  }

  return (configured || DEFAULT_LOCAL_API).replace(/\/$/, '');
}

const API_BASE_URL = resolveApiBaseUrl();

export { API_BASE_URL };
