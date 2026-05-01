import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import ThirdFactPage from '../src/pages/ThirdFactPage';

describe('ThirdFactPage', () => {
  it('renders the public Third Fact manifesto with claim discipline', () => {
    render(<ThirdFactPage />);

    expect(screen.getByRole('heading', { level: 1, name: /BIZRA/i })).toBeInTheDocument();
    expect(screen.getAllByText(/Humanity is not the fuel/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Claim Discipline Active/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/No Unverified Technical Claims/i).length).toBeGreaterThan(0);
  });

  it('keeps cited evidence links visible and accessible', () => {
    render(<ThirdFactPage />);

    expect(screen.getAllByRole('link', { name: /UNCTAD/i })[0]).toHaveAttribute('href', expect.stringContaining('unctad.org'));
    expect(screen.getAllByRole('link', { name: /IEA · Energy and AI/i })[0]).toHaveAttribute('href', expect.stringContaining('iea.org'));
    expect(screen.getByRole('navigation', { name: /Section navigation/i })).toBeInTheDocument();
  });

  it('sets Third Fact document metadata for the SPA route', () => {
    render(<ThirdFactPage />);

    expect(document.title).toBe('BIZRA — The Third Fact: Humanity Is the Infrastructure');
    expect(document.querySelector('meta[name="description"]')?.getAttribute('content')).toContain('humanity is not the fuel');
  });
});
