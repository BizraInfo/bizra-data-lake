; BIZRA Ghost Overlay — AutoHotkey v2 Renderer
; Frosted-glass HUD that projects proactive suggestions over any active window.
;
; Standing on Giants: Fitts (target acquisition) · Norman (affordance + feedback)
;                     Gibson (ecological perception) · Engelbart (augmented intellect)
;
; Architecture:
;   Node0 (Python) → Ghost WS Bridge (:9743) → [this script] → WS_EX_LAYERED overlay
;
; Prerequisites:
;   - AutoHotkey v2.0+ installed on Windows host
;   - Ghost WS Bridge running on localhost:9743 (or WSL IP)
;   - Config values from config/proactive_config.yaml (loaded below)
;
; Usage:
;   Start-Process "scripts\ghost_overlay.ahk"          ; PowerShell
;   autohotkey.exe scripts\ghost_overlay.ahk            ; CMD
;
; Created: 2026-02-25 | BIZRA Ghost Overlay v0.1

#Requires AutoHotkey v2.0
#SingleInstance Force
Persistent

; ---------------------------------------------------------------------------
; Configuration (from environment, never hardcoded)
; ---------------------------------------------------------------------------

WS_HOST := EnvGet("GHOST_WS_HOST") || "127.0.0.1"
WS_PORT := EnvGet("GHOST_WS_PORT") || "9743"
WS_URL  := "ws://" . WS_HOST . ":" . WS_PORT . "/ws/ghost"

IDLE_TIMEOUT_MS     := Integer(EnvGet("GHOST_IDLE_TIMEOUT_MS") || "5000")
MAX_SUGGESTIONS     := 3

; Sovereign Gesture hotkeys (user-configurable via env)
HOTKEY_SOLIDIFY := EnvGet("GHOST_HOTKEY_SOLIDIFY") || "#+g"   ; Win+Shift+G
HOTKEY_DISMISS  := EnvGet("GHOST_HOTKEY_DISMISS")  || "Escape"
HOTKEY_NEXT     := EnvGet("GHOST_HOTKEY_NEXT")     || "#+Down"
HOTKEY_PREV     := EnvGet("GHOST_HOTKEY_PREV")     || "#+Up"

; Design tokens (BIZRA brand — Sovereign Darkness palette)
BG_COLOR     := "080E1B"     ; rgba(8,14,27) — near-black
GOLD_COLOR   := "C9A962"     ; BIZRA gold accent
TEXT_COLOR   := "F8F4EC"     ; Off-white for readability
MUTED_COLOR  := "6E6A60"     ; Muted text
PASS_COLOR   := "2EB86A"     ; Ihsan pass — green
BLOCK_COLOR  := "C93A4A"     ; Ihsan blocked — red
PENDING_COLOR := "C9A962"    ; Ihsan pending — gold

; Overlay dimensions
OVERLAY_WIDTH  := 380
CARD_HEIGHT    := 72
HEADER_HEIGHT  := 36
FOOTER_HEIGHT  := 28
BORDER_RADIUS  := 12

; ---------------------------------------------------------------------------
; State
; ---------------------------------------------------------------------------

global overlayVisible := false
global suggestions := []
global activeIndex := 0
global dismissTimer := 0
global overlayGui := ""
global wsConnected := false

; ---------------------------------------------------------------------------
; Overlay GUI Creation
; ---------------------------------------------------------------------------

CreateOverlayGui() {
    global overlayGui

    ; Create layered, topmost, tool window (no taskbar entry)
    overlayGui := Gui("+AlwaysOnTop +ToolWindow -Caption +E0x80000")  ; WS_EX_LAYERED
    overlayGui.BackColor := BG_COLOR
    overlayGui.MarginX := 12
    overlayGui.MarginY := 8

    ; Make semi-transparent
    WinSetTransparent(225, overlayGui)  ; 88% opacity ≈ 225/255

    return overlayGui
}

; ---------------------------------------------------------------------------
; Render Overlay Content
; ---------------------------------------------------------------------------

RenderOverlay(suggestionsArr, posX := "", posY := "") {
    global overlayGui, overlayVisible, suggestions, activeIndex, dismissTimer

    suggestions := suggestionsArr
    activeIndex := 0

    ; Destroy previous overlay if exists
    if overlayVisible {
        try overlayGui.Destroy()
    }

    if (suggestions.Length = 0)
        return

    ; Create fresh overlay
    overlayGui := CreateOverlayGui()

    ; Header
    overlayGui.SetFont("s11 q5", "Segoe UI")
    overlayGui.AddText("c" . GOLD_COLOR . " w" . (OVERLAY_WIDTH - 24), "Sovereign Suggestion")

    ; Suggestion cards
    Loop Min(suggestions.Length, MAX_SUGGESTIONS) {
        idx := A_Index
        s := suggestions[idx]
        isActive := (idx = activeIndex + 1)

        ; Card background highlight for active
        if isActive {
            overlayGui.AddText("c" . GOLD_COLOR . " w" . (OVERLAY_WIDTH - 24) . " Section", "▸ " . s.action_label)
        } else {
            overlayGui.AddText("c" . TEXT_COLOR . " w" . (OVERLAY_WIDTH - 24) . " Section", "  " . s.action_label)
        }

        ; Intent summary line
        overlayGui.SetFont("s9 q5", "Consolas")
        overlayGui.AddText("c" . MUTED_COLOR . " xs+16 w" . (OVERLAY_WIDTH - 48), s.intent_summary)

        ; Ihsan badge
        if (s.ihsan_precheck = "pass") {
            badgeText := "✓ Ihsan " . Format("{:.2f}", s.ihsan_score)
            overlayGui.AddText("c" . PASS_COLOR . " xs+16 w200", badgeText)
        } else if (s.ihsan_precheck = "blocked") {
            badgeText := "✗ BLOCKED"
            overlayGui.AddText("c" . BLOCK_COLOR . " xs+16 w200", badgeText)
            if s.HasProp("block_reason") && s.block_reason {
                overlayGui.AddText("c" . BLOCK_COLOR . " xs+32 w200", s.block_reason)
            }
        } else {
            overlayGui.AddText("c" . PENDING_COLOR . " xs+16 w200", "⏳ Pending")
        }

        ; Reset font for next card
        overlayGui.SetFont("s11 q5", "Segoe UI")
    }

    ; Footer with gesture hints
    overlayGui.SetFont("s9 q5", "Segoe UI")
    overlayGui.AddText("c" . MUTED_COLOR . " w" . (OVERLAY_WIDTH - 24), "Win+Shift+G to act  ·  Esc to dismiss")

    ; Position near cursor or specified coordinates
    if (posX = "" || posY = "") {
        CoordMode("Mouse", "Screen")
        MouseGetPos(&mx, &my)
        posX := mx + 20
        posY := my + 20
    }

    ; Ensure overlay stays on screen
    monW := SysGet(78)  ; SM_CXVIRTUALSCREEN
    monH := SysGet(79)  ; SM_CYVIRTUALSCREEN
    if (posX + OVERLAY_WIDTH > monW)
        posX := monW - OVERLAY_WIDTH - 10
    if (posY + 300 > monH)
        posY := posY - 320

    overlayGui.Show("x" . posX . " y" . posY . " NoActivate")
    overlayVisible := true

    ; Start auto-dismiss timer
    if dismissTimer
        SetTimer(DismissOverlay, 0)
    dismissTimer := SetTimer(DismissOverlay, -IDLE_TIMEOUT_MS)
}

; ---------------------------------------------------------------------------
; Dismiss / Hide
; ---------------------------------------------------------------------------

DismissOverlay(*) {
    global overlayVisible, overlayGui, dismissTimer

    if overlayVisible {
        try overlayGui.Destroy()
        overlayVisible := false
    }
    if dismissTimer {
        SetTimer(DismissOverlay, 0)
        dismissTimer := 0
    }
}

; ---------------------------------------------------------------------------
; Sovereign Gestures
; ---------------------------------------------------------------------------

SolidifyAction(*) {
    global suggestions, activeIndex, overlayVisible

    if !overlayVisible || suggestions.Length = 0
        return

    active := suggestions[activeIndex + 1]

    if (active.ihsan_precheck = "pass") {
        ; Dispatch action via Desktop Bridge JSON-RPC
        DispatchToDesktopBridge(active.ahk_action_id, active)
        ; Gold flash feedback
        FlashOverlayBorder()
        DismissOverlay()
    } else if (active.ihsan_precheck = "blocked") {
        ; Show veto explanation — don't dismiss
        ToolTip("Ihsan Gate: " . (active.HasProp("block_reason") ? active.block_reason : "Blocked"), , , 2)
        SetTimer(() => ToolTip(,,,2), -3000)
    }
}

ScrollNext(*) {
    global activeIndex, suggestions, overlayVisible
    if !overlayVisible || suggestions.Length = 0
        return
    activeIndex := Mod(activeIndex + 1, Min(suggestions.Length, MAX_SUGGESTIONS))
    ; Re-render with updated active index
    RenderOverlay(suggestions)
}

ScrollPrev(*) {
    global activeIndex, suggestions, overlayVisible
    if !overlayVisible || suggestions.Length = 0
        return
    activeIndex := Mod(activeIndex - 1 + Min(suggestions.Length, MAX_SUGGESTIONS), Min(suggestions.Length, MAX_SUGGESTIONS))
    RenderOverlay(suggestions)
}

; ---------------------------------------------------------------------------
; Desktop Bridge Dispatch (JSON-RPC over TCP :9742)
; ---------------------------------------------------------------------------

DispatchToDesktopBridge(actionId, suggestion) {
    try {
        payload := '{"jsonrpc":"2.0","method":"dispatch_action","params":{'
            . '"action_id":"' . actionId . '",'
            . '"channel":"Ahk",'
            . '"permit_scope":"ghost_overlay"'
            . '},"id":1}'

        ; TCP send to Desktop Bridge on localhost:9742
        socket := ComObject("MSWinsock.Winsock")
        socket.RemoteHost := "127.0.0.1"
        socket.RemotePort := 9742
        socket.Connect()
        socket.SendData(payload)
        socket.Close()
    } catch as err {
        ; Graceful degradation — log but don't crash
        ToolTip("Bridge unavailable: " . err.Message, , , 3)
        SetTimer(() => ToolTip(,,,3), -3000)
    }
}

; ---------------------------------------------------------------------------
; Flash Effect (gold border pulse on successful dispatch)
; ---------------------------------------------------------------------------

FlashOverlayBorder() {
    global overlayGui
    if !overlayVisible
        return
    ; Brief gold flash via transparency change
    WinSetTransparent(255, overlayGui)
    SetTimer(() => (overlayVisible ? WinSetTransparent(225, overlayGui) : ""), -300)
    SetTimer(() => (overlayVisible ? WinSetTransparent(180, overlayGui) : ""), -600)
}

; ---------------------------------------------------------------------------
; WebSocket Client (simplified polling via HTTP — full WS requires library)
; ---------------------------------------------------------------------------

; NOTE: AHK v2 does not have native WebSocket support.
; For the prototype, we poll the /health endpoint and accept events via
; a lightweight HTTP bridge. In production, use WebSocket.ahk library
; or an Electron thin-client that forwards WS messages to this AHK script
; via named pipes or WM_COPYDATA.
;
; For now, the overlay can also be triggered via:
;   1. Named pipe from Node0 Python process
;   2. WM_COPYDATA messages from a companion WS→AHK bridge process
;   3. File-based trigger (Node0 writes JSON, AHK watches file)

PollHealthEndpoint() {
    global wsConnected
    try {
        whr := ComObject("WinHttp.WinHttpRequest.5.1")
        whr.Open("GET", "http://" . WS_HOST . ":" . WS_PORT . "/health", true)
        whr.Send()
        whr.WaitForResponse(2)
        if (whr.Status = 200) {
            wsConnected := true
            return true
        }
    } catch {
        wsConnected := false
    }
    return false
}

; ---------------------------------------------------------------------------
; File-based Event Trigger (prototype bridge)
; ---------------------------------------------------------------------------
; Node0 writes overlay events to this JSON file. AHK watches it.

TRIGGER_FILE := A_ScriptDir . "\..\sovereign_state\ghost_overlay_trigger.json"

WatchTriggerFile() {
    global TRIGGER_FILE

    if !FileExist(TRIGGER_FILE)
        return

    try {
        content := FileRead(TRIGGER_FILE, "UTF-8")
        FileDelete(TRIGGER_FILE)  ; Consume the trigger

        ; Parse JSON (simplified — AHK v2 has no native JSON parser)
        ; In production, use Jxon library or similar
        if InStr(content, '"show_overlay"') {
            ; Extract suggestions from JSON (basic parsing for prototype)
            parsedSuggestions := ParseOverlayJson(content)
            if parsedSuggestions.Length > 0
                RenderOverlay(parsedSuggestions)
        } else if InStr(content, '"dismiss_overlay"') {
            DismissOverlay()
        }
    } catch as err {
        ; Silent failure — trigger file may be partially written
    }
}

ParseOverlayJson(jsonStr) {
    ; Minimal JSON extraction for prototype
    ; Production should use Jxon.ahk library
    result := []

    ; For prototype: create a sample suggestion from the trigger
    s := {}
    s.action_label := "Proactive suggestion"
    s.intent_summary := "From Node0 prediction"
    s.ihsan_precheck := "pending"
    s.ihsan_score := 0.0
    s.ahk_action_id := "act_prototype"

    ; Try to extract action_label if present
    if RegExMatch(jsonStr, '"action_label"\s*:\s*"([^"]+)"', &m)
        s.action_label := m[1]
    if RegExMatch(jsonStr, '"intent_summary"\s*:\s*"([^"]+)"', &m)
        s.intent_summary := m[1]
    if RegExMatch(jsonStr, '"ihsan_precheck"\s*:\s*"([^"]+)"', &m)
        s.ihsan_precheck := m[1]
    if RegExMatch(jsonStr, '"ihsan_score"\s*:\s*([0-9.]+)', &m)
        s.ihsan_score := Float(m[1])
    if RegExMatch(jsonStr, '"ahk_action_id"\s*:\s*"([^"]+)"', &m)
        s.ahk_action_id := m[1]

    result.Push(s)
    return result
}

; ---------------------------------------------------------------------------
; Register Hotkeys
; ---------------------------------------------------------------------------

Hotkey(HOTKEY_SOLIDIFY, SolidifyAction)
Hotkey(HOTKEY_DISMISS, DismissOverlay, "On")  ; Always active while overlay visible
Hotkey(HOTKEY_NEXT, ScrollNext)
Hotkey(HOTKEY_PREV, ScrollPrev)

; ---------------------------------------------------------------------------
; Main Loop — Poll for events
; ---------------------------------------------------------------------------

; Initial health check
if PollHealthEndpoint() {
    ToolTip("Ghost Overlay connected to WS Bridge", , , 1)
    SetTimer(() => ToolTip(,,,1), -2000)
} else {
    ToolTip("Ghost Overlay: WS Bridge not detected (file trigger mode)", , , 1)
    SetTimer(() => ToolTip(,,,1), -3000)
}

; File-based trigger watcher (500ms poll)
SetTimer(WatchTriggerFile, 500)

; Periodic health check (30s)
SetTimer(PollHealthEndpoint, 30000)

; Tray icon
A_IconTip := "BIZRA Ghost Overlay v0.1"
TraySetIcon("Shell32.dll", 44)  ; Ghost-like icon

; Keep script running
return
