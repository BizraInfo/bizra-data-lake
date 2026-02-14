; ============================================================================
; BIZRA Desktop Bridge Client — AutoHotkey v2
;
; Minimal 3-hotkey command surface for the BIZRA Sovereign Runtime.
; Connects to desktop_bridge.py via TCP JSON-RPC on 127.0.0.1:9742.
;
; Hotkeys:
;   Win+B          -> ping (tooltip: "BIZRA: Connected | Uptime: Xs")
;   Ctrl+B, Q      -> sovereign_query (input dialog -> show result)
;   Ctrl+B, S      -> status (tooltip: Node0 status summary)
;
; Standing on Giants:
;   Boyd (OODA): Fast observe-orient-decide-act loop
;   Shannon (SNR): Signal first, bandwidth later
;
; Created: 2026-02-13 | BIZRA Desktop Bridge v1.0
; ============================================================================

#Requires AutoHotkey v2.0
#SingleInstance Force
Persistent

; ---------------------------------------------------------------------------
; Configuration
; ---------------------------------------------------------------------------

global BRIDGE_HOST := "127.0.0.1"
global BRIDGE_PORT := 9742
global MAX_RECONNECT_DELAY := 30000  ; ms
global TOOLTIP_DURATION := 3000      ; ms
global BRIDGE_TOKEN := EnvGet("BIZRA_BRIDGE_TOKEN")

; Connection state
global gSocket := 0
global gConnected := false
global gReconnectDelay := 1000  ; ms, doubles on failure (1s -> 2s -> 4s -> ... -> 30s)

; ---------------------------------------------------------------------------
; Winsock2 helpers (ws2_32.dll)
; ---------------------------------------------------------------------------

WS2_Init() {
    ; WSAStartup — initialize Winsock
    wsaData := Buffer(408, 0)
    result := DllCall("ws2_32\WSAStartup", "UShort", 0x0202, "Ptr", wsaData.Ptr, "Int")
    if (result != 0) {
        MsgBox("WSAStartup failed: " result, "BIZRA Bridge Error")
        return false
    }
    return true
}

WS2_Connect(host, port) {
    ; Create socket (AF_INET=2, SOCK_STREAM=1, IPPROTO_TCP=6)
    sock := DllCall("ws2_32\socket", "Int", 2, "Int", 1, "Int", 6, "Ptr")
    if (sock = -1 || sock = 0) {
        return 0
    }

    ; Resolve host to address
    addr := DllCall("ws2_32\inet_addr", "AStr", host, "UInt")
    if (addr = 0xFFFFFFFF) {
        DllCall("ws2_32\closesocket", "Ptr", sock)
        return 0
    }

    ; Build sockaddr_in struct
    sockAddr := Buffer(16, 0)
    NumPut("Short", 2, sockAddr, 0)           ; sin_family = AF_INET
    NumPut("UShort", WS2_Htons(port), sockAddr, 2)  ; sin_port (network byte order)
    NumPut("UInt", addr, sockAddr, 4)          ; sin_addr

    ; Connect
    result := DllCall("ws2_32\connect", "Ptr", sock, "Ptr", sockAddr.Ptr, "Int", 16, "Int")
    if (result != 0) {
        DllCall("ws2_32\closesocket", "Ptr", sock)
        return 0
    }

    return sock
}

WS2_Htons(val) {
    return ((val & 0xFF) << 8) | ((val >> 8) & 0xFF)
}

WS2_Send(sock, text) {
    ; Append newline delimiter
    msg := text "`n"
    buf := Buffer(StrPut(msg, "UTF-8") - 1, 0)
    StrPut(msg, buf.Ptr, buf.Size, "UTF-8")
    sent := DllCall("ws2_32\send", "Ptr", sock, "Ptr", buf.Ptr, "Int", buf.Size, "Int", 0, "Int")
    return sent > 0
}

WS2_Recv(sock, timeout_ms := 5000) {
    ; Set receive timeout via setsockopt (SOL_SOCKET=0xFFFF, SO_RCVTIMEO=0x1006)
    tv := Buffer(4, 0)
    NumPut("UInt", timeout_ms, tv, 0)
    DllCall("ws2_32\setsockopt", "Ptr", sock, "Int", 0xFFFF, "Int", 0x1006, "Ptr", tv.Ptr, "Int", 4)

    ; Read into buffer
    recvBuf := Buffer(65536, 0)
    received := DllCall("ws2_32\recv", "Ptr", sock, "Ptr", recvBuf.Ptr, "Int", recvBuf.Size, "Int", 0, "Int")

    if (received <= 0) {
        return ""
    }

    return StrGet(recvBuf.Ptr, received, "UTF-8")
}

WS2_Close(sock) {
    if (sock != 0) {
        DllCall("ws2_32\closesocket", "Ptr", sock)
    }
}

WS2_Cleanup() {
    DllCall("ws2_32\WSACleanup")
}

; ---------------------------------------------------------------------------
; JSON-RPC helpers
; ---------------------------------------------------------------------------

JsonEscape(text) {
    escaped := StrReplace(text, "\", "\\")
    escaped := StrReplace(escaped, '"', '\"')
    return escaped
}

BridgeTimestampMs() {
    ; Unix epoch milliseconds
    return DateDiff("19700101000000", A_NowUTC, "Seconds") * 1000
}

BridgeNonce() {
    return Format("{:016X}{:08X}{:08X}", A_TickCount, Random(0, 0x7FFFFFFF), Random(0, 0x7FFFFFFF))
}

BuildRequest(method, params := "", id := 1) {
    global BRIDGE_TOKEN

    if (BRIDGE_TOKEN = "") {
        return ""
    }

    tsMs := BridgeTimestampMs()
    nonce := BridgeNonce()
    tokenEsc := JsonEscape(BRIDGE_TOKEN)
    headers := '{"X-BIZRA-TOKEN":"' tokenEsc '","X-BIZRA-TS":' tsMs ',"X-BIZRA-NONCE":"' nonce '"}'

    if (params = "") {
        return '{"jsonrpc":"2.0","method":"' method '","headers":' headers ',"id":' id '}'
    }
    return '{"jsonrpc":"2.0","method":"' method '","params":' params ',"headers":' headers ',"id":' id '}'
}

; Minimal JSON value extractor (no dependency on JSON library).
; Extracts the value for a given key from a flat JSON string.
JsonGet(json, key) {
    needle := '"' key '":'
    pos := InStr(json, needle)
    if (!pos)
        return ""

    start := pos + StrLen(needle)

    ; Skip whitespace
    while (SubStr(json, start, 1) = " " || SubStr(json, start, 1) = "`t")
        start++

    ch := SubStr(json, start, 1)

    if (ch = '"') {
        ; String value — find closing quote
        end := InStr(json, '"',, start + 1)
        return SubStr(json, start + 1, end - start - 1)
    }
    if (ch = '{' || ch = '[') {
        ; Object/Array — find matching close bracket (naive: first occurrence)
        close := (ch = '{') ? '}' : ']'
        depth := 1
        i := start + 1
        while (i <= StrLen(json) && depth > 0) {
            c := SubStr(json, i, 1)
            if (c = ch)
                depth++
            else if (c = close)
                depth--
            i++
        }
        return SubStr(json, start, i - start)
    }

    ; Number, bool, null — read until comma, }, or ]
    end := start
    while (end <= StrLen(json)) {
        c := SubStr(json, end, 1)
        if (c = ',' || c = '}' || c = ']' || c = '`n')
            break
        end++
    }
    return Trim(SubStr(json, start, end - start))
}

; ---------------------------------------------------------------------------
; Connection management
; ---------------------------------------------------------------------------

ConnectBridge() {
    global gSocket, gConnected, gReconnectDelay

    if (gConnected && gSocket != 0) {
        return true
    }

    gSocket := WS2_Connect(BRIDGE_HOST, BRIDGE_PORT)
    if (gSocket = 0) {
        gConnected := false
        return false
    }

    gConnected := true
    gReconnectDelay := 1000  ; Reset backoff on success
    UpdateTrayTip("Connected")
    return true
}

DisconnectBridge() {
    global gSocket, gConnected
    if (gSocket != 0) {
        WS2_Close(gSocket)
        gSocket := 0
    }
    gConnected := false
    UpdateTrayTip("Disconnected")
}

SendCommand(method, params := "") {
    global gSocket, gConnected, gReconnectDelay

    ; Try connect if not connected
    if (!gConnected) {
        if (!ConnectBridge()) {
            ShowTooltip("BIZRA: Not connected (retrying in " (gReconnectDelay // 1000) "s)")
            ; Schedule reconnect with backoff
            SetTimer(TryReconnect, -gReconnectDelay)
            gReconnectDelay := Min(gReconnectDelay * 2, MAX_RECONNECT_DELAY)
            return ""
        }
    }

    reqId := A_TickCount
    request := BuildRequest(method, params, reqId)
    if (request = "") {
        ShowTooltip("BIZRA: Missing BIZRA_BRIDGE_TOKEN")
        return ""
    }

    if (!WS2_Send(gSocket, request)) {
        ; Send failed — connection likely dead
        DisconnectBridge()
        ShowTooltip("BIZRA: Connection lost, reconnecting...")
        SetTimer(TryReconnect, -gReconnectDelay)
        gReconnectDelay := Min(gReconnectDelay * 2, MAX_RECONNECT_DELAY)
        return ""
    }

    response := WS2_Recv(gSocket, 10000)
    if (response = "") {
        DisconnectBridge()
        ShowTooltip("BIZRA: No response, reconnecting...")
        SetTimer(TryReconnect, -gReconnectDelay)
        gReconnectDelay := Min(gReconnectDelay * 2, MAX_RECONNECT_DELAY)
        return ""
    }

    return response
}

TryReconnect() {
    ConnectBridge()
}

; ---------------------------------------------------------------------------
; UI helpers
; ---------------------------------------------------------------------------

ShowTooltip(text, duration := 0) {
    if (duration = 0)
        duration := TOOLTIP_DURATION
    ToolTip(text)
    SetTimer(ClearTooltip, -duration)
}

ClearTooltip() {
    ToolTip()
}

UpdateTrayTip(status) {
    TraySetIcon("Shell32.dll", (status = "Connected") ? 44 : 132)
    A_IconTip := "BIZRA Bridge: " status
}

; ---------------------------------------------------------------------------
; Hotkeys
; ---------------------------------------------------------------------------

; Win+B -> ping
#b:: {
    response := SendCommand("ping")
    if (response = "") {
        return
    }

    status := JsonGet(response, "status")
    uptime := JsonGet(response, "uptime_s")

    if (status = "" && InStr(response, '"error"')) {
        errMsg := JsonGet(response, "message")
        ShowTooltip("BIZRA: Error - " errMsg)
        return
    }

    ShowTooltip("BIZRA: " status " | Uptime: " uptime "s")
}

; Ctrl+B, Q -> sovereign_query
^b:: {
    ; Wait for 'q' key within 2 seconds
    ih := InputHook("L1 T2")
    ih.Start()
    ih.Wait()
    key := ih.Input

    if (key = "q" || key = "Q") {
        DoSovereignQuery()
    } else if (key = "s" || key = "S") {
        DoStatus()
    } else if (key = "i" || key = "I") {
        DoInvokeSkill()
    } else if (key = "l" || key = "L") {
        DoListSkills()
    }
}

DoSovereignQuery() {
    query := InputBox("Enter your query:", "BIZRA Sovereign Query", "W400 H120")
    if (query.Result != "OK" || query.Value = "") {
        return
    }

    ; Escape double quotes in query for JSON
    escaped := StrReplace(query.Value, '"', '\"')
    escaped := StrReplace(escaped, '\', '\\')
    params := '{"query":"' escaped '"}'

    ShowTooltip("BIZRA: Querying...")
    response := SendCommand("sovereign_query", params)
    if (response = "") {
        return
    }

    content := JsonGet(response, "content")
    errField := JsonGet(response, "error")
    latency := JsonGet(response, "latency_ms")

    if (content != "") {
        model := JsonGet(response, "model")
        MsgBox(content "`n`n[Model: " model " | Latency: " latency "ms]",
               "BIZRA Sovereign Response")
    } else if (errField != "") {
        MsgBox("Error: " errField, "BIZRA Query Error")
    } else {
        MsgBox("Unexpected response:`n" response, "BIZRA Bridge")
    }
}

DoStatus() {
    response := SendCommand("status")
    if (response = "") {
        return
    }

    node := JsonGet(response, "node")
    uptime := JsonGet(response, "bridge_uptime_s")
    reqCount := JsonGet(response, "request_count")
    fateGate := JsonGet(response, "fate_gate")
    inference := JsonGet(response, "inference")

    ; Build summary
    summary := "Node: " node
    summary .= "`nUptime: " uptime "s"
    summary .= "`nRequests: " reqCount
    summary .= "`nFATE Gate: " fateGate
    summary .= "`nInference: " inference

    ShowTooltip(summary, 5000)
}

DoInvokeSkill() {
    skill := InputBox("Skill name:", "BIZRA Invoke Skill", "W400 H120")
    if (skill.Result != "OK" || skill.Value = "") {
        return
    }

    ; Escape skill name for JSON
    escapedSkill := StrReplace(skill.Value, '"', '\"')
    params := '{"skill":"' escapedSkill '"}'

    ; Optional: ask for JSON inputs
    inputs := InputBox("Inputs (JSON, leave empty to skip):", "BIZRA Skill Inputs", "W400 H120")
    if (inputs.Result = "OK" && inputs.Value != "") {
        params := '{"skill":"' escapedSkill '","inputs":' inputs.Value '}'
    }

    ShowTooltip("BIZRA: Invoking " skill.Value "...")
    response := SendCommand("invoke_skill", params)
    if (response = "") {
        return
    }

    output := JsonGet(response, "output")
    errField := JsonGet(response, "error")
    receiptId := JsonGet(response, "receipt_id")

    if (output != "") {
        MsgBox(output "`n`n[Receipt: " receiptId "]",
               "BIZRA Skill: " skill.Value)
    } else if (errField != "") {
        MsgBox("Error: " errField, "BIZRA Skill Error")
    } else {
        MsgBox("Unexpected response:`n" response, "BIZRA Bridge")
    }
}

DoListSkills() {
    response := SendCommand("list_skills")
    if (response = "") {
        return
    }

    count := JsonGet(response, "count")
    errField := JsonGet(response, "error")

    if (errField != "") {
        ShowTooltip("BIZRA: " errField, 3000)
        return
    }

    ; Build summary of available skills
    summary := "Available Skills (" count "):`n"

    ; Extract skills array content (naive — shows raw JSON for now)
    skills := JsonGet(response, "skills")
    if (skills != "") {
        ; Show first 500 chars of skills list
        display := SubStr(skills, 1, 500)
        summary .= display
    } else {
        summary .= "(none loaded)"
    }

    ShowTooltip(summary, 6000)
}

; ---------------------------------------------------------------------------
; Initialization
; ---------------------------------------------------------------------------

; Initialize Winsock
if (!WS2_Init()) {
    MsgBox("Failed to initialize Winsock2", "BIZRA Bridge Fatal Error")
    ExitApp
}

; Set tray icon and tip
UpdateTrayTip("Disconnected")

; Attempt initial connection
ConnectBridge()

; Cleanup on exit
OnExit(ExitHandler)
ExitHandler(reason, code) {
    DisconnectBridge()
    WS2_Cleanup()
}
