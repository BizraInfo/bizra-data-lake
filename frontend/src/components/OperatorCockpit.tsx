// BIZRA Operator Cockpit — Sprint 3 Product Surface
// See /mnt/user-data/outputs/BIZRA_Operator_Cockpit.jsx for the full component
// This marker file indicates the cockpit is delivered as a Claude artifact
// and ready to be wired into the frontend build system.
//
// Location: frontend/src/components/OperatorCockpit.jsx (334 lines)
// Dependencies: React (useState, useEffect)
// Design: Dark sovereign (#020408), Cinzel headers, JetBrains Mono data
// Data source: WebSocket to kernel_daemon.py:9740 (to be wired)
//
// The component shows the full constitutional pipeline:
// Intent → Guardian → Execution → Receipt → Chain → Evidence
//
// Sprint 3 Epic 2: Wire to live EventBus data via WebSocket.
export { default } from './OperatorCockpit.impl';
