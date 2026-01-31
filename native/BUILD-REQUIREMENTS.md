# BIZRA Native Build Requirements

## Prerequisites

### Required for Rust FATE Binding

1. **Visual Studio Build Tools 2022** (or full Visual Studio 2022)
   - Download: https://visualstudio.microsoft.com/downloads/
   - Select: "Desktop development with C++" workload
   - Or minimum: MSVC v143 - VS 2022 C++ x64/x86 build tools

2. **Rust** (already installed)
   - Verified: cargo 1.92.0, rustc 1.92.0

### Quick Install (PowerShell Admin)

```powershell
# Install VS Build Tools with C++ workload
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --add Microsoft.VisualStudio.Workload.VCTools"
```

## Build Commands

### FATE Binding (Rust → TypeScript via napi-rs)

```powershell
cd native/fate-binding

# Install Node dependencies
npm install

# Build native addon (requires VS Build Tools)
npm run build

# Or use cargo directly
cargo build --release
```

### Iceoryx2 Bridge (Rust IPC)

```powershell
cd native/iceoryx-bridge
cargo build --release
```

## Troubleshooting

### Error: `link.exe not found`
**Solution**: Install Visual Studio Build Tools with C++ workload

### Error: `dlltool.exe not found` (GNU toolchain)
**Solution**: Use MSVC toolchain instead: `rustup default stable-x86_64-pc-windows-msvc`

### Error: `z3.h not found`
**Solution**: Install Z3 solver
```powershell
# Using vcpkg
vcpkg install z3:x64-windows

# Or download from https://github.com/Z3Prover/z3/releases
```

## Integration Status

| Component | Build Status | Integration |
|-----------|--------------|-------------|
| FATE Binding | ⏳ Needs VS Build Tools | TypeScript via napi-rs |
| Iceoryx2 Bridge | ⏳ Needs VS Build Tools | Python sandbox IPC |
| TypeScript Core | ✅ Ready | Built with tsconfig.json |
| Python Core | ✅ Verified | Tests passing |
