import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';

// Lazy-loaded phases need direct imports for testing
import Splash from '../src/phases/Splash';
import Genesis from '../src/phases/Genesis';
import TeachSteps from '../src/phases/TeachSteps';

describe('Splash Phase', () => {
  it('renders BIZRA branding', () => {
    render(<Splash onStart={vi.fn()} />);
    expect(screen.getByText('BIZRA')).toBeTruthy();
    expect(screen.getByText('SOVEREIGN AI OPERATING SYSTEM')).toBeTruthy();
  });

  it('calls onStart when button clicked', () => {
    const onStart = vi.fn();
    render(<Splash onStart={onStart} />);
    const btn = screen.getByText('INITIALIZE NODE');
    fireEvent.click(btn);
    expect(onStart).toHaveBeenCalledOnce();
  });
});

describe('Genesis Phase', () => {
  it('renders identity input', () => {
    render(<Genesis onDone={vi.fn()} />);
    expect(screen.getByPlaceholderText('Your sovereign name')).toBeTruthy();
    expect(screen.getByText('IDENTITY GENESIS')).toBeTruthy();
  });

  it('disables generate button when name is empty', () => {
    render(<Genesis onDone={vi.fn()} />);
    const btn = screen.getByText('GENERATE IDENTITY');
    expect(btn).toBeDisabled();
  });

  it('enables generate button when name is entered', () => {
    render(<Genesis onDone={vi.fn()} />);
    const input = screen.getByPlaceholderText('Your sovereign name');
    fireEvent.change(input, { target: { value: 'Sovereign' } });
    const btn = screen.getByText('GENERATE IDENTITY');
    expect(btn).not.toBeDisabled();
  });

  it('restores initialName', () => {
    render(<Genesis initialName="Restored" onDone={vi.fn()} />);
    const input = screen.getByPlaceholderText('Your sovereign name') as HTMLInputElement;
    expect(input.value).toBe('Restored');
  });

  it('calls onNameChange on input', () => {
    const onNameChange = vi.fn();
    render(<Genesis onNameChange={onNameChange} onDone={vi.fn()} />);
    const input = screen.getByPlaceholderText('Your sovereign name');
    fireEvent.change(input, { target: { value: 'Node1' } });
    expect(onNameChange).toHaveBeenCalledWith('Node1');
  });
});

describe('TeachSteps Phase', () => {
  it('renders first question', () => {
    render(<TeachSteps onDone={vi.fn()} />);
    expect(screen.getByText('What is your typical work schedule?')).toBeTruthy();
    expect(screen.getByText(/STEP 1 OF/)).toBeTruthy();
  });

  it('renders progress pips', () => {
    const { container } = render(<TeachSteps onDone={vi.fn()} />);
    // 5 questions = 5 progress pips (divs with specific widths)
    const pips = container.querySelectorAll('div[style*="border-radius: 99"]');
    expect(pips.length).toBe(5);
  });

  it('calls onDraftChange when interacting', () => {
    const onDraftChange = vi.fn();
    render(<TeachSteps onDraftChange={onDraftChange} onDone={vi.fn()} />);
    const input = screen.getByPlaceholderText('8:00-18:00');
    fireEvent.change(input, { target: { value: '9-17' } });
    expect(onDraftChange).toHaveBeenCalled();
  });
});
