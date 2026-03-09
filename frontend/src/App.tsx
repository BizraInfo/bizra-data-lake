import { lazy, Suspense, useEffect, useState } from 'react';
import { usePerformance } from './hooks/usePerformance';
import { EMPTY_TEACH_DRAFT, loadAppSession, saveAppSession } from './lib/persistence';
import type { AppSessionState } from './lib/persistence';

const TrustSite = lazy(() => import('./phases/TrustSite'));
const Splash = lazy(() => import('./phases/Splash'));
const Genesis = lazy(() => import('./phases/Genesis'));
const TeachSteps = lazy(() => import('./phases/TeachSteps'));
const Assembly = lazy(() => import('./phases/Assembly'));
const Dashboard = lazy(() => import('./phases/Dashboard'));

function LoadingFallback() {
  return (
    <div style={{
      minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: 4, color: 'var(--gold)',
    }}>
      INITIALIZING...
    </div>
  );
}

export function App() {
  const [session, setSession] = useState<AppSessionState>(() => loadAppSession());

  usePerformance(import.meta.env.PROD);

  useEffect(() => {
    saveAppSession(session);
  }, [session]);

  const handleEnter = () => {
    setSession(prev => ({ ...prev, phase: 'splash' }));
  };

  const handleIdentityDraftChange = (pendingIdentityName: string) => {
    setSession(prev => ({ ...prev, pendingIdentityName }));
  };

  const handleGenesisDone = (userName: string) => {
    setSession(prev => ({
      ...prev,
      userName,
      pendingIdentityName: userName,
      phase: 'teach',
    }));
  };

  const handleTeachDraftChange = (teachDraft: AppSessionState['teachDraft']) => {
    setSession(prev => ({ ...prev, teachDraft }));
  };

  const handleTeachDone = (config: AppSessionState['config']) => {
    setSession(prev => ({
      ...prev,
      config,
      teachDraft: EMPTY_TEACH_DRAFT,
      phase: 'assembly',
    }));
  };

  const handleAssemblyDone = () => {
    setSession(prev => ({ ...prev, phase: 'dashboard' }));
  };

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)' }}>
      <div className="geo-bg" />
      <div className="grain" />
      <Suspense fallback={<LoadingFallback />}>
        {session.phase === 'trust' && <TrustSite onEnter={handleEnter} />}
        {session.phase === 'splash' && <Splash onStart={() => setSession(prev => ({ ...prev, phase: 'genesis' }))} />}
        {session.phase === 'genesis' && (
          <Genesis
            initialName={session.pendingIdentityName}
            onNameChange={handleIdentityDraftChange}
            onDone={handleGenesisDone}
          />
        )}
        {session.phase === 'teach' && (
          <TeachSteps
            initialDraft={session.teachDraft}
            onDraftChange={handleTeachDraftChange}
            onDone={handleTeachDone}
          />
        )}
        {session.phase === 'assembly' && (
          <Assembly userName={session.userName} config={session.config} onDone={handleAssemblyDone} />
        )}
        {session.phase === 'dashboard' && <Dashboard userName={session.userName} config={session.config} />}
      </Suspense>
    </div>
  );
}
