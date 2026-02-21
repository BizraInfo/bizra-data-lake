; ============================================================
; BIZRA AHK Bridge Server — JSON-RPC 2.0 over TCP
; ============================================================
; Listens on 127.0.0.1:9742 (configurable via BIZRA_BRIDGE_PORT)
; Implements: invoke_skill, actuator_execute
; Auth: X-BIZRA-TOKEN + timestamp + nonce replay protection
;
; Standing on Giants: General Magic (Telescript, 1994)
; artifact: bizra-node/src/action_executor.rs:351-410
; ============================================================

#Requires AutoHotkey v2.0
#SingleInstance Force
Persistent

; --- Configuration -----------------------------------------------------------
global CONFIG := LoadConfig()
global BRIDGE_TOKEN := EnvGet("BIZRA_BRIDGE_TOKEN")
global NONCE_CACHE := Map()
global NONCE_WINDOW_MS := 60000
global MAX_PAYLOAD_LEN := 4096
global ACTIVE_CONNECTIONS := 0
global MAX_CONNECTIONS := 8
global RUNNING := true

; Blocked patterns for Guardian veto (case-insensitive matching)
global BLOCKED_PATTERNS := [
    "rm -rf",
    "format c:",
    "del /f /s",
    "shutdown /s",
    "shutdown /r",
    "taskkill /f",
    "bypass safety",
    "override guardian"
]

; --- Startup -----------------------------------------------------------------
if (BRIDGE_TOKEN = "") {
    LogError("BIZRA_BRIDGE_TOKEN not set. Exiting.")
    ExitApp 1
}

PORT := EnvGet("BIZRA_BRIDGE_PORT")
if (PORT = "")
    PORT := CONFIG.Has("port") ? CONFIG["port"] : 9742

LogInfo("BIZRA AHK Bridge starting on 127.0.0.1:" PORT)
LogInfo("Skills directory: " CONFIG["skills_dir"])
LogInfo("Run allowed: " (CONFIG.Has("allow_run") ? CONFIG["allow_run"] : "false"))

; Start TCP server
try {
    server := SocketServer("127.0.0.1", PORT)
    LogInfo("AHK Bridge listening on 127.0.0.1:" PORT)
} catch as err {
    LogError("Failed to bind port " PORT ": " err.Message)
    LogError("Check if another bridge instance is running.")
    ExitApp 1
}

OnExit((*) => Cleanup(server))
return

; --- Socket Server Class -----------------------------------------------------
class SocketServer {
    __New(host, port) {
        this.socket := Socket()
        this.socket.Bind(host, port)
        this.socket.Listen(MAX_CONNECTIONS)
        this.socket.OnAccept := ObjBindMethod(this, "OnAccept")
    }

    OnAccept(socket) {
        global ACTIVE_CONNECTIONS
        if (ACTIVE_CONNECTIONS >= MAX_CONNECTIONS) {
            LogWarn("Connection limit reached (" MAX_CONNECTIONS "), rejecting")
            socket.Disconnect()
            return
        }
        ACTIVE_CONNECTIONS++
        try {
            HandleConnection(socket)
        } catch as err {
            LogError("Connection handler error: " err.Message)
        }
        ACTIVE_CONNECTIONS--
    }

    __Delete() {
        try this.socket.Disconnect()
    }
}

; --- Connection Handler ------------------------------------------------------
HandleConnection(conn) {
    line := conn.RecvLine(5000)  ; 5s read timeout
    if (line = "") {
        conn.Disconnect()
        return
    }

    ; Parse JSON
    try {
        request := Jxon_Load(&line)
    } catch {
        SendJsonRpcError(conn, -32700, "Parse error", "null")
        conn.Disconnect()
        return
    }

    id := request.Has("id") ? request["id"] : "null"

    ; Authentication gate
    authResult := Authenticate(request)
    if (authResult != "") {
        SendJsonRpcError(conn, -32600, authResult, id)
        conn.Disconnect()
        return
    }

    ; Method dispatch
    method := request.Has("method") ? request["method"] : ""
    params := request.Has("params") ? request["params"] : Map()

    switch method {
        case "invoke_skill":
            result := HandleInvokeSkill(params)
        case "actuator_execute":
            result := HandleActuatorExecute(params)
        default:
            SendJsonRpcError(conn, -32601, "Method not found: " method, id)
            conn.Disconnect()
            return
    }

    ; Send response
    if (result.Has("error")) {
        SendJsonRpcError(conn, result["error_code"], result["error"], id)
    } else {
        SendJsonRpcResult(conn, result["result"], id)
    }

    conn.Disconnect()
}

; --- Authentication ----------------------------------------------------------
Authenticate(request) {
    global BRIDGE_TOKEN, NONCE_CACHE, NONCE_WINDOW_MS

    if (!request.Has("headers"))
        return "Missing headers"

    headers := request["headers"]

    ; Token check
    token := headers.Has("X-BIZRA-TOKEN") ? headers["X-BIZRA-TOKEN"] : ""
    if (token != BRIDGE_TOKEN)
        return "Invalid token"

    ; Timestamp check (30s window)
    ts := headers.Has("X-BIZRA-TS") ? headers["X-BIZRA-TS"] : 0
    nowMs := A_TickCount  ; Approximate — use DateDiff for real ms
    if (Abs(nowMs - ts) > 30000)
        return "Timestamp outside 30s window"

    ; Nonce replay check
    nonce := headers.Has("X-BIZRA-NONCE") ? headers["X-BIZRA-NONCE"] : ""
    if (nonce = "")
        return "Missing nonce"

    ; Prune expired nonces
    PruneNonceCache()

    if (NONCE_CACHE.Has(nonce))
        return "Nonce replay detected"

    NONCE_CACHE[nonce] := A_TickCount
    return ""  ; Auth OK
}

PruneNonceCache() {
    global NONCE_CACHE, NONCE_WINDOW_MS
    now := A_TickCount
    expired := []
    for nonce, ts in NONCE_CACHE {
        if (now - ts > NONCE_WINDOW_MS)
            expired.Push(nonce)
    }
    for _, nonce in expired {
        NONCE_CACHE.Delete(nonce)
    }
}

; --- invoke_skill Handler ----------------------------------------------------
HandleInvokeSkill(params) {
    skill := params.Has("skill") ? params["skill"] : ""
    if (skill = "")
        return ErrorResult(-32602, "params.skill is required")

    inputs := params.Has("inputs") ? params["inputs"] : Map()

    ; Sanitize skill name (prevent path traversal)
    safeSkill := RegExReplace(skill, "[/\\\.]{2,}|[^\w\-]", "")
    if (safeSkill != skill)
        return ErrorResult(-32602, "Invalid skill name")

    skillsDir := CONFIG["skills_dir"]
    skillPath := skillsDir "\" safeSkill ".ahk"

    if (!FileExist(skillPath))
        return ErrorResult(-32602, "Skill not found: " skill)

    ; Write inputs to temp file
    tempFile := A_Temp "\bizra_skill_" A_TickCount ".json"
    try {
        inputsJson := Jxon_Dump(&inputs)
        FileAppend(inputsJson, tempFile, "UTF-8")
    } catch {
        return ErrorResult(-32000, "Failed to write skill input")
    }

    ; Execute skill script with timeout
    try {
        output := RunSkillWithTimeout(skillPath, tempFile, 10000)
    } catch as err {
        try FileDelete(tempFile)
        return ErrorResult(-32000, "Skill error: " err.Message)
    }

    try FileDelete(tempFile)
    return OkResult(output)
}

RunSkillWithTimeout(scriptPath, inputFile, timeoutMs) {
    cmd := '"' A_AhkPath '" /script "' scriptPath '" "' inputFile '"'
    shell := ComObject("WScript.Shell")
    exec := shell.Exec(cmd)

    startTick := A_TickCount
    while (!exec.Status) {
        if (A_TickCount - startTick > timeoutMs) {
            exec.Terminate()
            throw Error("Skill timeout after " timeoutMs "ms")
        }
        Sleep(50)
    }

    output := exec.StdOut.ReadAll()
    if (exec.ExitCode != 0) {
        errOutput := exec.StdErr.ReadAll()
        throw Error("Skill exited with code " exec.ExitCode ": " errOutput)
    }

    return Trim(output)
}

; --- actuator_execute Handler ------------------------------------------------
HandleActuatorExecute(params) {
    global MAX_PAYLOAD_LEN, BLOCKED_PATTERNS

    code := params.Has("code") ? params["code"] : ""
    intent := params.Has("intent") ? params["intent"] : "execute"
    targetApp := params.Has("target_app") ? params["target_app"] : ""

    ; --- Safety Guards (FR-5) ---

    ; Payload length check
    if (StrLen(code) > MAX_PAYLOAD_LEN)
        return ErrorResult(-32001, "Guardian veto: payload too large (" StrLen(code) " > " MAX_PAYLOAD_LEN ")")

    ; Blocked pattern check
    codeLower := StrLower(code)
    for _, pattern in BLOCKED_PATTERNS {
        if InStr(codeLower, pattern)
            return ErrorResult(-32001, "Guardian veto: blocked pattern '" pattern "'")
    }

    ; Target app allowlist check
    if (CONFIG.Has("app_allowlist") && CONFIG["app_allowlist"] != "") {
        allowed := StrSplit(CONFIG["app_allowlist"], ",")
        found := false
        for _, app in allowed {
            if (Trim(app) = targetApp) {
                found := true
                break
            }
        }
        if (!found && targetApp != "")
            return ErrorResult(-32001, "Guardian veto: target_app '" targetApp "' not in allowlist")
    }

    ; --- Intent Dispatch (FR-4) ---
    switch StrLower(intent) {
        case "click":
            try {
                Click(code)
                return OkResult("click executed")
            } catch as err {
                return ErrorResult(-32000, "Click failed: " err.Message)
            }

        case "type":
            try {
                ; Escape special AHK characters
                safeCode := AhkEscapeSend(code)
                Send(safeCode)
                return OkResult("type executed")
            } catch as err {
                return ErrorResult(-32000, "Type failed: " err.Message)
            }

        case "execute":
            ; Most dangerous — extra config gate
            allowRun := CONFIG.Has("allow_run") ? CONFIG["allow_run"] : "false"
            if (allowRun != "true")
                return ErrorResult(-32001, "Guardian veto: Run disabled in config (allow_run=false)")

            try {
                Run(code)
                return OkResult("run executed")
            } catch as err {
                return ErrorResult(-32000, "Run failed: " err.Message)
            }

        case "read":
            try {
                ; Extract text from window control
                if (targetApp != "") {
                    text := ControlGetText(code, targetApp)
                } else {
                    text := WinGetText(code)
                }
                return OkResult(text)
            } catch as err {
                return ErrorResult(-32000, "Read failed: " err.Message)
            }

        default:
            return ErrorResult(-32602, "Unknown intent: " intent)
    }
}

; --- Utility: Escape AHK Send special chars ----------------------------------
AhkEscapeSend(text) {
    ; Escape characters that have special meaning in Send
    text := StrReplace(text, "{", "{{}")
    text := StrReplace(text, "}", "{}}")
    text := StrReplace(text, "^", "{^}")
    text := StrReplace(text, "!", "{!}")
    text := StrReplace(text, "+", "{+}")
    text := StrReplace(text, "#", "{#}")
    return text
}

; --- JSON-RPC Response Helpers -----------------------------------------------
OkResult(value) {
    result := Map()
    result["result"] := value
    return result
}

ErrorResult(code, message) {
    result := Map()
    result["error"] := message
    result["error_code"] := code
    return result
}

SendJsonRpcResult(conn, resultValue, id) {
    response := Map()
    response["jsonrpc"] := "2.0"
    response["result"] := Map("result", resultValue)
    response["id"] := id
    json := Jxon_Dump(&response)
    conn.SendLine(json)
}

SendJsonRpcError(conn, code, message, id) {
    response := Map()
    response["jsonrpc"] := "2.0"
    response["error"] := Map("code", code, "message", message)
    response["id"] := id
    json := Jxon_Dump(&response)
    conn.SendLine(json)
}

; --- Configuration Loader ----------------------------------------------------
LoadConfig() {
    config := Map()
    config["port"] := 9742
    config["skills_dir"] := A_ScriptDir "\skills"
    config["allow_run"] := "false"
    config["app_allowlist"] := ""

    iniPath := A_ScriptDir "\bridge_config.ini"
    if FileExist(iniPath) {
        try {
            config["port"] := IniRead(iniPath, "bridge", "port", 9742)
            config["skills_dir"] := IniRead(iniPath, "bridge", "skills_dir", config["skills_dir"])
            config["allow_run"] := IniRead(iniPath, "bridge", "allow_run", "false")
            config["app_allowlist"] := IniRead(iniPath, "bridge", "app_allowlist", "")
        }
    }

    ; Ensure skills directory exists
    if !DirExist(config["skills_dir"])
        DirCreate(config["skills_dir"])

    return config
}

; --- Logging -----------------------------------------------------------------
LogInfo(msg) {
    OutputDebug("[BIZRA-BRIDGE INFO] " FormatTime(, "yyyy-MM-dd HH:mm:ss") " " msg)
}

LogWarn(msg) {
    OutputDebug("[BIZRA-BRIDGE WARN] " FormatTime(, "yyyy-MM-dd HH:mm:ss") " " msg)
}

LogError(msg) {
    OutputDebug("[BIZRA-BRIDGE ERROR] " FormatTime(, "yyyy-MM-dd HH:mm:ss") " " msg)
}

; --- Cleanup -----------------------------------------------------------------
Cleanup(server) {
    global RUNNING
    RUNNING := false
    LogInfo("AHK Bridge shutting down")
    try server.__Delete()
}
