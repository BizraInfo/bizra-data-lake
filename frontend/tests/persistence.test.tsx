import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { loadAppSession, loadMissionSession, saveAppSession, saveMissionSession } from '../src/lib/persistence';
import type { AppSessionState } from '../src/lib/persistence';
import Genesis from '../src/phases/Genesis';
import TeachSteps from '../src/phases/TeachSteps';

describe('frontend persistence', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it('round-trips app session state through storage', () => {
    const session: AppSessionState = {
      phase: 'dashboard',
      userName: 'Mumo',
      pendingIdentityName: 'Mumo',
      config: { communication_pref: 'Detailed explanations' },
      teachDraft: {
        step: 2,
        answers: { work_schedule: '8:00-18:00' },
        textValue: 'ignored on restore',
        selected: ['Engineering'],
      },
    };

    saveAppSession(session);

    expect(loadAppSession()).toEqual(session);
  });

  it('drops corrupt mission storage and falls back to defaults', () => {
    window.localStorage.setItem('bizra.ddagi.mission.v1.mumo', '{bad json');

    const session = loadMissionSession('mumo');

    expect(session.messages).toEqual([]);
    expect(session.nodeState.rac).toBe(0);
  });

  it('restores a persisted mission session for the current user', () => {
    saveMissionSession('mumo', {
      messages: [{ agent: 'NEXUS', text: 'Restored', type: 'greet', ts: 1 }],
      nodeState: {
        seed: 2,
        bloom: 0.5,
        rac: 3,
        vac: 3,
        tier: 0,
        mye: 0,
        s1: 0,
        s2: 3,
        streak: 3,
        ihsan: 0.97,
        reflexes: 0,
        legendary: 0,
        epic: 1,
        sovereignty: 0.42,
      },
    });

    const session = loadMissionSession('mumo');

    expect(session.messages[0]?.text).toBe('Restored');
    expect(session.nodeState.seed).toBe(2);
    expect(session.nodeState.sovereignty).toBe(0.42);
  });

  it('hydrates genesis name draft from persisted session props', () => {
    render(<Genesis initialName="Sovereign Name" onDone={vi.fn()} />);

    expect(screen.getByDisplayValue('Sovereign Name')).toBeInTheDocument();
  });

  it('resumes teach flow from the stored draft step', () => {
    render(
      <TeachSteps
        initialDraft={{
          step: 1,
          answers: {},
          textValue: '',
          selected: ['Chrome'],
        }}
        onDone={vi.fn()}
      />,
    );

    expect(screen.getByText(/STEP 2 OF/i)).toBeInTheDocument();
    expect(screen.getByText('Chrome')).toBeInTheDocument();
  });
});