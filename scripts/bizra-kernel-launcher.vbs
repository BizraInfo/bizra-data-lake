' ═══════════════════════════════════════════════════════════════
'  BIZRA SOVEREIGN KERNEL — Silent OS Boot Launcher v2.0
'  Runs on Windows startup. No console flash. No user action.
'
'  v2: Starts the kernel DAEMON (HTTP server on :9740) which
'      handles routing, backend management, and health checks.
'      Falls back to direct HTML if Python unavailable.
' ═══════════════════════════════════════════════════════════════

Set objShell = CreateObject("WScript.Shell")
Set objFSO   = CreateObject("Scripting.FileSystemObject")

' ── Resolve paths ──
Dim scriptDir, bizraRoot, daemonPath, pidPath

scriptDir  = objFSO.GetParentFolderName(WScript.ScriptFullName)
bizraRoot  = objFSO.GetParentFolderName(scriptDir)
daemonPath = bizraRoot & "\core\sovereign\kernel_daemon.py"
pidPath    = bizraRoot & "\sovereign_state\kernel.pid"

' ── Check if daemon already running ──
If objFSO.FileExists(pidPath) Then
    ' PID file exists — daemon might be running. Open browser to it.
    objShell.Run "http://127.0.0.1:9740/", 1, False
    GoTo CleanExit
End If

' ── Find Python (prefer pythonw for no-console) ──
Dim pythonExe
pythonExe = ""

' 1. Project venv pythonw
Dim venvPythonW
venvPythonW = bizraRoot & "\.venv\Scripts\pythonw.exe"
If objFSO.FileExists(venvPythonW) Then
    pythonExe = venvPythonW
End If

' 2. Project venv python
If pythonExe = "" Then
    Dim venvPython
    venvPython = bizraRoot & "\.venv\Scripts\python.exe"
    If objFSO.FileExists(venvPython) Then
        pythonExe = venvPython
    End If
End If

' 3. System pythonw
If pythonExe = "" Then
    On Error Resume Next
    Dim sysPythonW
    sysPythonW = objShell.ExpandEnvironmentStrings("%LOCALAPPDATA%") & "\Programs\Python\Python311\pythonw.exe"
    If objFSO.FileExists(sysPythonW) Then pythonExe = sysPythonW
    On Error GoTo 0
End If

' 4. System python via PATH
If pythonExe = "" Then
    On Error Resume Next
    objShell.Run "pythonw --version", 0, True
    If Err.Number = 0 Then pythonExe = "pythonw"
    On Error GoTo 0
End If

' ── Start daemon if Python found ──
If pythonExe <> "" And objFSO.FileExists(daemonPath) Then
    ' Start kernel daemon (no console window)
    Dim cmd
    cmd = """" & pythonExe & """ """ & daemonPath & """"
    objShell.CurrentDirectory = bizraRoot
    objShell.Run cmd, 0, False

    ' Wait briefly then open browser
    WScript.Sleep 2500
    objShell.Run "http://127.0.0.1:9740/", 1, False
Else
    ' ── Fallback: open HTML directly (no daemon) ──
    Dim statePath, targetPath
    statePath = bizraRoot & "\sovereign_state\kernel_initialized.json"

    If objFSO.FileExists(statePath) Then
        targetPath = bizraRoot & "\frontend\public\terminal-emulator.html"
    Else
        targetPath = bizraRoot & "\frontend\public\bizra-installer.html"
    End If

    If objFSO.FileExists(targetPath) Then
        objShell.Run """" & targetPath & """", 1, False
    Else
        MsgBox "BIZRA kernel files not found." & vbCrLf & "Expected: " & targetPath, vbCritical, "BIZRA Sovereign Kernel"
    End If
End If

CleanExit:
Set objFSO   = Nothing
Set objShell = Nothing
WScript.Quit 0
