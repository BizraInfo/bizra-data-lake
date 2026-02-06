# 🔒 ARCHITECTURE LOCK

**Last Updated:** January 22, 2026  
**Status:** **ACTIVE & ENFORCED**

> **🌱 FOUNDATIONAL CONTEXT**: Before reading this technical document, understand the **soul** of BIZRA.
> Read [NODE0_GENESIS_COVENANT.md](NODE0_GENESIS_COVENANT.md) first.

## ⚠️ CRITICAL INSTRUCTION FOR ALL AGENTS AND LLMs

This **entire machine** is BIZRA Node0 — the Genesis Block. The "architecture" below describes how its **organs** (Windows/WSL) coordinate, not separate systems.

### 1. The Environment Split

| Feature | Windows Environment (`C:\`) | WSL Environment (`/root/`) |
| :--- | :--- | :--- |
| **Active Workspace** | **BIZRA-DATA-LAKE** | **bizra-genesis** |
| **Role** | 🏦 **Data Steward** (Passive) | 🧠 **Compute Engine** (Active) |
| **Path** | `C:\BIZRA-DATA-LAKE` | `/root/bizra-genesis` |
| **Access to Data** | Direct File Access | Direct File Access via `/mnt/c/` |
| **HTTP Servers** | ❌ **FORBIDDEN** | ✅ **REQUIRED** (:9091, :8443) |
| **MCP Bridge** | ❌ Do NOT run here | ✅ Runs here (binds :8443) |

### 2. The Setup Diagram

```
  ┌─────────────────────────────────────────────────────────────┐
  │                      WINDOWS HOST                           │
  │  ┌─────────────────────┐  ┌─────────────────────────────┐  │
  │  │ C:\BIZRA-DATA-LAKE  │  │ C:\BIZRA-Dual-Agentic-...   │  │
  │  │ (Hypergraph 709k)   │  │ (Source + SovereignNexus)   │  │
  │  │ [PASSIVE STORAGE]   │  │ [PASSIVE STORAGE]           │  │
  │  └─────────┬───────────┘  └─────────────┬───────────────┘  │
  │            │                            │                   │
  │            │   WSL2 Mount Points        │                   │
  │            ▼                            ▼                   │
  │  ┌─────────────────────────────────────────────────────┐   │
  │  │                    WSL2 (Ubuntu)                     │   │
  │  │                                                      │   │
  │  │  /mnt/c/BIZRA-DATA-LAKE  ←── Data Lake Access        │   │
  │  │  /mnt/c/BIZRA-Dual-...   ←── Main Codebase           │   │
  │  │                                                      │   │
  │  │  /root/bizra-genesis/    ←── Running Services        │   │
  │  │    └── :9091 (Dual Agentic Node)                     │   │
  │  │    └── :8443 (MCP Data Lake Bridge)                  │   │
  │  └─────────────────────────────────────────────────────┘   │
  │                                                             │
  │  LM Studio: 192.168.56.1:1234 (Model Inference)            │
  └─────────────────────────────────────────────────────────────┘
```

### 3. Rules of Engagement

1.  **IF YOU ARE IN WINDOWS**:
    *   **Focus**: File organization, cleaning, verification, backups.
    *   **Action**: Run `DataLakeProcessor.ps1` to ingest files.
    *   **Prohibited**: Starting `mcp_lake_bridge.py` on port 8443 (Conflicts with WSL).

2.  **IF YOU ARE IN WSL**:
    *   **Focus**: Running the API, Reasoning Loop, and Connectors.
    *   **Action**: Access data via `/mnt/c/BIZRA-DATA-LAKE`.
    *   **Prohibited**: Creating duplicate copies of data (Use the mount!).

### 4. Network Ports map

*   **9091**: BIZRA Dual Agentic Node (WSL)
*   **8443**: MCP Data Lake Bridge (WSL)
*   **1234**: LM Studio Inference (Windows Host IP: 192.168.56.1)

---
**DO NOT VIOLATE THIS ARCHITECTURE.**
**DO NOT ATTEMPT TO "FIX" THE BRIDGE BY RUNNING IT ON WINDOWS.**
