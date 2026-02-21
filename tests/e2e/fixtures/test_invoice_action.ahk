; tests/e2e/fixtures/test_invoice_action.ahk
; ============================================================
; BIZRA Action E2E fixture
; Opens Notepad, types a deterministic sentinel payload, exits.
; ============================================================

#Requires AutoHotkey v2.0

sentinel := "BIZRA_TEST_INVOICE_SENTINEL_2026"

Run "notepad.exe"
if !WinWaitActive("ahk_exe notepad.exe",, 3) {
    ExitApp 2
}

SendText sentinel
Sleep 200

; Return control by closing Notepad without save prompt persistence.
WinClose "ahk_exe notepad.exe"
if WinWaitActive("Notepad",, 1) {
    ; "Don't Save" accelerator on modern Notepad
    Send "!n"
}

ExitApp 0

