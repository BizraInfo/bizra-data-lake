/**
 * Auth state hook — manages token + user identity.
 * Persists to localStorage; exposes login/logout/register.
 */

import { useCallback, useEffect, useState } from 'react';
import { api } from '../lib/api';

interface AuthState {
  token: string | null;
  nodeId: string | null;
  name: string | null;
  loading: boolean;
}

const STORAGE_KEY = 'bizra_auth';

function loadStored(): Pick<AuthState, 'token' | 'nodeId' | 'name'> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    // corrupted storage — ignore
  }
  return { token: null, nodeId: null, name: null };
}

export function useAuth() {
  const [state, setState] = useState<AuthState>(() => ({
    ...loadStored(),
    loading: false,
  }));

  useEffect(() => {
    if (state.token) {
      api.setToken(state.token);
    } else {
      api.clearToken();
    }
  }, [state.token]);

  const persist = useCallback((token: string, nodeId: string, name: string) => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ token, nodeId, name }));
    api.setToken(token);
    setState({ token, nodeId, name, loading: false });
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    setState(s => ({ ...s, loading: true }));
    const res = await api.login({ username, password });
    persist(res.token, res.node_id, username);
  }, [persist]);

  const register = useCallback(async (username: string, password: string, name: string) => {
    setState(s => ({ ...s, loading: true }));
    const res = await api.register({ username, password, name });
    persist(res.token, res.node_id, name);
  }, [persist]);

  const logout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    api.clearToken();
    setState({ token: null, nodeId: null, name: null, loading: false });
  }, []);

  return {
    ...state,
    authenticated: !!state.token,
    login,
    register,
    logout,
  };
}
