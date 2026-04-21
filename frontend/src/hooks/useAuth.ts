/**
 * Auth state hook — manages access_token + user identity.
 * Persists to localStorage; exposes login/logout/register.
 *
 * Backend contract: core/sovereign/api.py `/v1/auth/{register,login,refresh}`.
 * Register requires email + accept_covenant; responses have nested
 * `tokens.access_token`. See `frontend/src/types.ts` for the wire types and
 * `tests/core/auth/test_auth_wire_contract.py` for the cross-layer guard.
 */

import { useCallback, useEffect, useState } from 'react';
import { api } from '../lib/api';

interface AuthState {
  token: string | null; // access_token from backend tokens.access_token
  userId: string | null; // backend user_id (flat on login, user.user_id on register)
  username: string | null;
  loading: boolean;
}

const STORAGE_KEY = 'bizra_auth';

function loadStored(): Pick<AuthState, 'token' | 'userId' | 'username'> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    // corrupted storage — ignore
  }
  return { token: null, userId: null, username: null };
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

  const persist = useCallback((token: string, userId: string, username: string) => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ token, userId, username }));
    api.setToken(token);
    setState({ token, userId, username, loading: false });
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    setState(s => ({ ...s, loading: true }));
    const res = await api.login({ username, password });
    persist(res.tokens.access_token, res.user_id, res.username);
  }, [persist]);

  const register = useCallback(
    async (username: string, email: string, password: string, acceptCovenant: boolean) => {
      setState(s => ({ ...s, loading: true }));
      const res = await api.register({
        username,
        email,
        password,
        accept_covenant: acceptCovenant,
      });
      persist(res.tokens.access_token, res.user.user_id, res.user.username);
    },
    [persist],
  );

  const logout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    api.clearToken();
    setState({ token: null, userId: null, username: null, loading: false });
  }, []);

  return {
    ...state,
    authenticated: !!state.token,
    login,
    register,
    logout,
  };
}
