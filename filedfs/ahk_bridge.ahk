; ============================================================
; BIZRA AHK Bridge Server — JSON-RPC 2.0 over TCP
; ============================================================
; Listens on 127.0.0.1:9742 (configurable via BIZRA_BRIDGE_PORT)
; Implements: invoke_skill, actuator_execute, get_context,
;   open_app, switch_window, type_text, click_element,
;   screenshot, read_clipboard, file_open, browser_navigate
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
        case "get_context":
            result := HandleGetContext(params)
        case "open_app":
            result := HandleOpenApp(params)
        case "switch_window":
            result := HandleSwitchWindow(params)
        case "type_text":
            result := HandleTypeText(params)
        case "click_element":
            result := HandleClickElement(params)
        case "screenshot":
            result := HandleScreenshot(params)
        case "read_clipboard":
            result := HandleReadClipboard(params)
        case "file_open":
            result := HandleFileOpen(params)
        case "browser_navigate":
            result := HandleBrowserNavigate(params)
        default:
            SendJsonRpcError(conn, -32601, "Method not found: " method, id)
            conn.Disconnect()
            return
    }

    ; Send response — pass full result Map to preserve perception-action metadata
    ; (pre_hash, post_hash, state_changed, outcome_confirmed) from HDA handlers
    if (result.Has("error")) {
        SendJsonRpcError(conn, result["error_code"], result["error"], id)
    } else {
        SendJsonRpcResult(conn, result, id)
    }

    conn.Disconnect()
}

; --- Epoch Milliseconds (cross-process compatible) --------------------------
; A_TickCount returns ms since boot — incompatible with Python's time.time()*1000.
; Use epoch ms for all timestamp comparisons so AHK and Python agree.
EpochMs() {
    ; Windows FILETIME: 100-nanosecond intervals since 1601-01-01
    ; Unix epoch: 1970-01-01 = FILETIME 116444736000000000
    ft := Buffer(8)
    DllCall("GetSystemTimeAsFileTime", "Ptr", ft)
    lo := NumGet(ft, 0, "UInt")
    hi := NumGet(ft, 4, "UInt")
    fileTime := (hi << 32) | lo
    ; Convert to milliseconds since Unix epoch
    return (fileTime - 116444736000000000) // 10000
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

    ; Timestamp check (120s window — matches Python AUTH_MAX_CLOCK_SKEW_MS)
    ts := headers.Has("X-BIZRA-TS") ? headers["X-BIZRA-TS"] : 0
    nowMs := EpochMs()
    if (Abs(nowMs - ts) > 120000)
        return "Timestamp outside 120s window"

    ; Nonce replay check
    nonce := headers.Has("X-BIZRA-NONCE") ? headers["X-BIZRA-NONCE"] : ""
    if (nonce = "")
        return "Missing nonce"

    ; Prune expired nonces
    PruneNonceCache()

    if (NONCE_CACHE.Has(nonce))
        return "Nonce replay detected"

    NONCE_CACHE[nonce] := EpochMs()
    return ""  ; Auth OK
}

PruneNonceCache() {
    global NONCE_CACHE, NONCE_WINDOW_MS
    now := EpochMs()
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

    ; --- Perception-Action Loop: capture pre-action screenshot ---
    actionLabel := "action_" A_TickCount
    preHash := CaptureScreenshotHash("pre_" actionLabel)

    ; --- Intent Dispatch (FR-4) ---
    actionOutcome := ""
    actionErr := ""

    switch StrLower(intent) {
        case "click":
            try {
                Click(code)
                actionOutcome := "click executed"
            } catch as err {
                actionErr := "Click failed: " err.Message
            }

        case "type":
            try {
                ; Escape special AHK characters
                safeCode := AhkEscapeSend(code)
                Send(safeCode)
                actionOutcome := "type executed"
            } catch as err {
                actionErr := "Type failed: " err.Message
            }

        case "execute":
            ; Most dangerous — extra config gate
            allowRun := CONFIG.Has("allow_run") ? CONFIG["allow_run"] : "false"
            if (allowRun != "true")
                return ErrorResult(-32001, "Guardian veto: Run disabled in config (allow_run=false)")

            try {
                Run(code)
                actionOutcome := "run executed"
            } catch as err {
                actionErr := "Run failed: " err.Message
            }

        case "read":
            try {
                ; Extract text from window control
                if (targetApp != "") {
                    text := ControlGetText(code, targetApp)
                } else {
                    text := WinGetText(code)
                }
                actionOutcome := text
            } catch as err {
                actionErr := "Read failed: " err.Message
            }

        default:
            return ErrorResult(-32602, "Unknown intent: " intent)
    }

    ; --- Perception-Action Loop: capture post-action screenshot ---
    ; Small delay to let the UI settle after the action
    Sleep(100)
    postHash := CaptureScreenshotHash("post_" actionLabel)

    ; Determine if the action visibly changed the screen
    stateChanged := (preHash != postHash)

    ; Read actions should NOT change state; mutating actions SHOULD
    if (StrLower(intent) = "read") {
        outcomeConfirmed := !stateChanged
    } else {
        outcomeConfirmed := stateChanged
    }

    ; Return error with verification metadata if action failed
    if (actionErr != "") {
        result := Map()
        result["error"] := actionErr
        result["error_code"] := -32000
        result["pre_hash"] := preHash
        result["post_hash"] := postHash
        result["state_changed"] := stateChanged
        result["outcome_confirmed"] := false
        return result
    }

    ; Build enriched result with perception-action verification
    result := Map()
    result["result"] := actionOutcome
    result["pre_hash"] := preHash
    result["post_hash"] := postHash
    result["state_changed"] := stateChanged
    result["outcome_confirmed"] := outcomeConfirmed
    return result
}

; --- Screenshot Capture (Phase 21 — Perception-Action Loop) -----------------
; Captures desktop state before/after actions, returning a SHA-256 hash for
; receipt chaining. Full bitmap stays on-disk; only the 64-char hex hash
; travels over JSON-RPC.  Closes the loop:
;   Agent decides -> AHK executes -> Screenshot verifies -> Receipt seals.
;
; Usage from actuator_execute:
;   preHash  := CaptureScreenshotHash("pre_" actionId)
;   ... execute action ...
;   postHash := CaptureScreenshotHash("post_" actionId)
;   stateChanged := (preHash != postHash)

CaptureScreenshotHash(label := "") {
    ; Build deterministic file path under temp
    screenshotDir := A_Temp "\bizra_screenshots"
    if !DirExist(screenshotDir)
        DirCreate(screenshotDir)

    timestamp := FormatTime(, "yyyyMMdd_HHmmss")
    safeLabel := RegExReplace(label, "[^\w\-]", "_")
    filePath := screenshotDir "\" timestamp "_" safeLabel ".bmp"

    ; Capture the full virtual screen (all monitors)
    try {
        ; Get virtual screen dimensions
        x := SysGet(76)   ; SM_XVIRTUALSCREEN
        y := SysGet(77)   ; SM_YVIRTUALSCREEN
        w := SysGet(78)   ; SM_CXVIRTUALSCREEN
        h := SysGet(79)   ; SM_CYVIRTUALSCREEN

        ; GDI+ bitmap capture via COM
        hDC := DllCall("GetDC", "Ptr", 0, "Ptr")
        hMemDC := DllCall("CreateCompatibleDC", "Ptr", hDC, "Ptr")
        hBitmap := DllCall("CreateCompatibleBitmap", "Ptr", hDC, "Int", w, "Int", h, "Ptr")
        hOld := DllCall("SelectObject", "Ptr", hMemDC, "Ptr", hBitmap, "Ptr")
        DllCall("BitBlt", "Ptr", hMemDC, "Int", 0, "Int", 0
            , "Int", w, "Int", h, "Ptr", hDC, "Int", x, "Int", y, "UInt", 0x00CC0020)
        DllCall("SelectObject", "Ptr", hMemDC, "Ptr", hOld)

        ; Save as BMP through GDI+ (init once — idempotent)
        static pToken := 0
        if (pToken = 0) {
            si := Buffer(24, 0)
            NumPut("UInt", 1, si, 0)
            DllCall("gdiplus\GdiplusStartup", "Ptr*", &pToken, "Ptr", si, "Ptr", 0)
        }
        pBitmapGdip := 0
        DllCall("gdiplus\GdipCreateBitmapFromHBITMAP", "Ptr", hBitmap, "Ptr", 0, "Ptr*", &pBitmapGdip)

        ; Use BMP CLSID {557cf400-1a04-11d3-9a73-0000f81ef32e}
        CLSID := Buffer(16)
        DllCall("ole32\CLSIDFromString", "Str", "{557CF400-1A04-11D3-9A73-0000F81EF32E}", "Ptr", CLSID)
        DllCall("gdiplus\GdipSaveImageToFile", "Ptr", pBitmapGdip, "Str", filePath, "Ptr", CLSID, "Ptr", 0)
        DllCall("gdiplus\GdipDisposeImage", "Ptr", pBitmapGdip)

        DllCall("DeleteObject", "Ptr", hBitmap)
        DllCall("DeleteDC", "Ptr", hMemDC)
        DllCall("ReleaseDC", "Ptr", 0, "Ptr", hDC)

    } catch as err {
        LogWarn("Screenshot capture failed: " err.Message)
        ; Fallback: hash the timestamp so callers always get a value
        return HashString("screenshot_error:" A_TickCount ":" label)
    }

    ; Hash the BMP file contents with SHA-256
    if FileExist(filePath) {
        hash := HashFile(filePath)
        LogInfo("Screenshot captured: " filePath " hash=" SubStr(hash, 1, 16) "...")
        return hash
    }

    return HashString("no_screenshot:" A_TickCount ":" label)
}

; SHA-256 of a file using Windows CNG (bcrypt.dll)
HashFile(path) {
    try {
        content := FileRead(path, "RAW")
        return HashBytes(content)
    } catch as err {
        return HashString("file_read_error:" err.Message)
    }
}

; SHA-256 of a string (UTF-8 encoded)
HashString(text) {
    buf := Buffer(StrPut(text, "UTF-8") - 1)
    StrPut(text, buf, "UTF-8")
    return HashBytes(buf)
}

; SHA-256 of raw bytes via Windows CNG (bcrypt.dll)
HashBytes(data) {
    ; Open algorithm provider
    hAlg := 0
    DllCall("bcrypt\BCryptOpenAlgorithmProvider"
        , "Ptr*", &hAlg, "Str", "SHA256", "Ptr", 0, "UInt", 0, "UInt")

    ; Get hash object size
    cbHashObj := 0
    cbData := 0
    DllCall("bcrypt\BCryptGetProperty"
        , "Ptr", hAlg, "Str", "ObjectLength"
        , "UInt*", &cbHashObj, "UInt", 4, "UInt*", &cbData, "UInt", 0, "UInt")

    ; Create hash object
    hashObj := Buffer(cbHashObj)
    hHash := 0
    DllCall("bcrypt\BCryptCreateHash"
        , "Ptr", hAlg, "Ptr*", &hHash, "Ptr", hashObj, "UInt", cbHashObj
        , "Ptr", 0, "UInt", 0, "UInt", 0, "UInt")

    ; Hash the data
    DllCall("bcrypt\BCryptHashData"
        , "Ptr", hHash, "Ptr", data, "UInt", data.Size, "UInt", 0, "UInt")

    ; Finalize — SHA-256 = 32 bytes
    digest := Buffer(32)
    DllCall("bcrypt\BCryptFinishHash"
        , "Ptr", hHash, "Ptr", digest, "UInt", 32, "UInt", 0, "UInt")

    ; Cleanup
    DllCall("bcrypt\BCryptDestroyHash", "Ptr", hHash, "UInt")
    DllCall("bcrypt\BCryptCloseAlgorithmProvider", "Ptr", hAlg, "UInt", 0, "UInt")

    ; Convert to hex string
    hex := ""
    loop 32
        hex .= Format("{:02x}", NumGet(digest, A_Index - 1, "UChar"))

    return hex
}

; === 8 Productized HDA Skills (Task 1.2) ====================================
; Each skill: validates params, captures pre/post hash, returns receipt-ready
; result with outcome_confirmed flag. Standing on Giants: General Magic (1994).
; ============================================================================

; --- open_app: Launch an application by name or path ------------------------
HandleOpenApp(params) {
    global BLOCKED_PATTERNS

    app := params.Has("app") ? params["app"] : ""
    if (app = "")
        return ErrorResult(-32602, "params.app is required")

    ; Guardian veto: block dangerous executables
    appLower := StrLower(app)
    for _, pattern in BLOCKED_PATTERNS {
        if InStr(appLower, pattern)
            return ErrorResult(-32001, "Guardian veto: blocked app '" app "'")
    }

    preHash := CaptureScreenshotHash("pre_open_app")

    try {
        Run(app)
        ; Wait briefly for app to appear
        Sleep(500)
    } catch as err {
        return ErrorResult(-32000, "Failed to open app: " err.Message)
    }

    postHash := CaptureScreenshotHash("post_open_app")
    stateChanged := (preHash != postHash)

    result := Map()
    result["result"] := "opened: " app
    result["pre_hash"] := preHash
    result["post_hash"] := postHash
    result["state_changed"] := stateChanged
    result["outcome_confirmed"] := stateChanged
    return result
}

; --- switch_window: Activate a window by title (partial match) --------------
HandleSwitchWindow(params) {
    title := params.Has("title") ? params["title"] : ""
    if (title = "")
        return ErrorResult(-32602, "params.title is required")

    preHash := CaptureScreenshotHash("pre_switch_window")

    try {
        if WinExist(title) {
            WinActivate(title)
            WinWaitActive(title, , 3)
            activeTitle := WinGetTitle("A")
        } else {
            return ErrorResult(-32000, "Window not found: " title)
        }
    } catch as err {
        return ErrorResult(-32000, "Switch window failed: " err.Message)
    }

    postHash := CaptureScreenshotHash("post_switch_window")
    stateChanged := (preHash != postHash)

    result := Map()
    result["result"] := "switched to: " activeTitle
    result["pre_hash"] := preHash
    result["post_hash"] := postHash
    result["state_changed"] := stateChanged
    result["outcome_confirmed"] := stateChanged
    return result
}

; --- type_text: Type text into the active window (safe-escaped) -------------
HandleTypeText(params) {
    text := params.Has("text") ? params["text"] : ""
    if (text = "")
        return ErrorResult(-32602, "params.text is required")

    ; Enforce max length to prevent clipboard bombing
    if (StrLen(text) > 2048)
        return ErrorResult(-32001, "Guardian veto: text too long (" StrLen(text) " > 2048)")

    preHash := CaptureScreenshotHash("pre_type_text")

    try {
        safeText := AhkEscapeSend(text)
        Send(safeText)
    } catch as err {
        return ErrorResult(-32000, "Type text failed: " err.Message)
    }

    Sleep(100)
    postHash := CaptureScreenshotHash("post_type_text")
    stateChanged := (preHash != postHash)

    result := Map()
    result["result"] := "typed " StrLen(text) " chars"
    result["pre_hash"] := preHash
    result["post_hash"] := postHash
    result["state_changed"] := stateChanged
    result["outcome_confirmed"] := stateChanged
    return result
}

; --- click_element: Click at coordinates or named control -------------------
HandleClickElement(params) {
    x := params.Has("x") ? params["x"] : ""
    y := params.Has("y") ? params["y"] : ""
    control := params.Has("control") ? params["control"] : ""
    button := params.Has("button") ? params["button"] : "left"

    if (x = "" && y = "" && control = "")
        return ErrorResult(-32602, "params.x+y or params.control required")

    preHash := CaptureScreenshotHash("pre_click")

    try {
        if (control != "") {
            ; Click a named control in the foreground window
            ControlClick(control, "A", , button)
        } else {
            ; Click at absolute coordinates
            Click(x " " y " " button)
        }
    } catch as err {
        return ErrorResult(-32000, "Click failed: " err.Message)
    }

    Sleep(100)
    postHash := CaptureScreenshotHash("post_click")
    stateChanged := (preHash != postHash)

    result := Map()
    result["result"] := control ? "clicked control: " control : "clicked at: " x "," y
    result["pre_hash"] := preHash
    result["post_hash"] := postHash
    result["state_changed"] := stateChanged
    result["outcome_confirmed"] := stateChanged
    return result
}

; --- screenshot: Capture current desktop state, return hash -----------------
HandleScreenshot(params) {
    label := params.Has("label") ? params["label"] : "manual_capture"

    hash := CaptureScreenshotHash(label)

    result := Map()
    result["result"] := Map("hash", hash, "label", label)
    result["outcome_confirmed"] := (hash != "")
    return result
}

; --- read_clipboard: Return clipboard hash + length (never raw content) -----
HandleReadClipboard(params) {
    includePlaintext := false
    if (params.Has("include_plaintext"))
        includePlaintext := params["include_plaintext"]

    try {
        clipText := A_Clipboard
        clipLen := StrLen(clipText)
        clipHash := ""
        clipPreview := ""

        if (clipText != "") {
            clipHash := HashString(clipText)
            ; Optionally include first 128 chars for preview (opt-in only)
            if (includePlaintext && clipLen > 0)
                clipPreview := SubStr(clipText, 1, 128)
        }

        result := Map()
        result["result"] := Map(
            "hash", clipHash,
            "length", clipLen,
            "preview", clipPreview,
            "has_content", clipLen > 0
        )
        result["outcome_confirmed"] := true
        return result
    } catch as err {
        return ErrorResult(-32000, "Read clipboard failed: " err.Message)
    }
}

; --- file_open: Open a file with its default application --------------------
HandleFileOpen(params) {
    global BLOCKED_PATTERNS

    path := params.Has("path") ? params["path"] : ""
    if (path = "")
        return ErrorResult(-32602, "params.path is required")

    ; Guardian veto: block dangerous paths
    pathLower := StrLower(path)
    for _, pattern in BLOCKED_PATTERNS {
        if InStr(pathLower, pattern)
            return ErrorResult(-32001, "Guardian veto: blocked path pattern '" pattern "'")
    }

    ; Verify file exists before opening
    if (!FileExist(path))
        return ErrorResult(-32000, "File not found: " path)

    preHash := CaptureScreenshotHash("pre_file_open")

    try {
        Run(path)
        Sleep(500)
    } catch as err {
        return ErrorResult(-32000, "File open failed: " err.Message)
    }

    postHash := CaptureScreenshotHash("post_file_open")
    stateChanged := (preHash != postHash)

    result := Map()
    result["result"] := "opened: " path
    result["pre_hash"] := preHash
    result["post_hash"] := postHash
    result["state_changed"] := stateChanged
    result["outcome_confirmed"] := stateChanged
    return result
}

; --- browser_navigate: Open URL in default browser --------------------------
HandleBrowserNavigate(params) {
    url := params.Has("url") ? params["url"] : ""
    if (url = "")
        return ErrorResult(-32602, "params.url is required")

    ; Only allow http/https URLs (prevent file:// and javascript: injection)
    if (!RegExMatch(url, "^https?://"))
        return ErrorResult(-32001, "Guardian veto: only http/https URLs allowed")

    ; Block known dangerous domains
    urlLower := StrLower(url)
    dangerousPatterns := ["javascript:", "data:", "vbscript:"]
    for _, pattern in dangerousPatterns {
        if InStr(urlLower, pattern)
            return ErrorResult(-32001, "Guardian veto: blocked URL pattern")
    }

    preHash := CaptureScreenshotHash("pre_browser_nav")

    try {
        Run(url)
        Sleep(1000)  ; Browser needs time to load
    } catch as err {
        return ErrorResult(-32000, "Browser navigate failed: " err.Message)
    }

    postHash := CaptureScreenshotHash("post_browser_nav")
    stateChanged := (preHash != postHash)

    result := Map()
    result["result"] := "navigated to: " url
    result["pre_hash"] := preHash
    result["post_hash"] := postHash
    result["state_changed"] := stateChanged
    result["outcome_confirmed"] := stateChanged
    return result
}

; --- get_context Handler (Task 1.1 — Live Desktop Context) ------------------
; Returns live desktop state: foreground window, process list, clipboard hash.
; Privacy-by-design: window titles are SHA-256 hashed by default.
; Plaintext requires explicit opt-in via params.plaintext_titles = true.
;
; Standing on Giants: Boyd (OODA observe phase), Shannon (information density)

HandleGetContext(params) {
    includePlaintext := false
    if (params.Has("plaintext_titles"))
        includePlaintext := params["plaintext_titles"]

    result := Map()

    ; --- 1. Foreground window info ---
    try {
        fgHwnd := WinGetID("A")
        fgTitle := WinGetTitle("A")
        fgClass := WinGetClass("A")
        fgPid := WinGetPID("A")
        fgProcess := WinGetProcessName("A")

        fg := Map()
        fg["title"] := includePlaintext ? fgTitle : HashString(fgTitle)
        fg["title_hashed"] := !includePlaintext
        fg["class"] := fgClass
        fg["process"] := fgProcess
        fg["pid"] := fgPid
        fg["hwnd"] := fgHwnd
        result["foreground"] := fg
    } catch as err {
        result["foreground"] := Map("error", "Could not detect foreground: " err.Message)
    }

    ; --- 2. Window list (visible windows only) ---
    try {
        windowList := []
        winIds := WinGetList()
        for _, id in winIds {
            try {
                title := WinGetTitle(id)
                ; Skip empty/invisible windows
                if (title = "" || !WinExist(id))
                    continue

                win := Map()
                win["title"] := includePlaintext ? title : HashString(title)
                win["title_hashed"] := !includePlaintext
                win["class"] := WinGetClass(id)
                win["process"] := WinGetProcessName(id)
                win["pid"] := WinGetPID(id)
                windowList.Push(win)

                ; Cap at 32 windows to avoid payload explosion
                if (windowList.Length >= 32)
                    break
            }
        }
        result["windows"] := windowList
        result["window_count"] := windowList.Length
    } catch as err {
        result["windows"] := []
        result["window_count"] := 0
        result["window_error"] := err.Message
    }

    ; --- 3. Clipboard hash (never send raw clipboard content) ---
    try {
        clipText := A_Clipboard
        if (clipText != "") {
            result["clipboard_hash"] := HashString(clipText)
            result["clipboard_length"] := StrLen(clipText)
        } else {
            result["clipboard_hash"] := ""
            result["clipboard_length"] := 0
        }
    } catch {
        result["clipboard_hash"] := ""
        result["clipboard_length"] := 0
    }

    ; --- 4. Screen geometry ---
    try {
        result["screen"] := Map(
            "width", SysGet(0),
            "height", SysGet(1),
            "virtual_width", SysGet(78),
            "virtual_height", SysGet(79),
            "monitor_count", MonitorGetCount()
        )
    } catch {
        result["screen"] := Map()
    }

    ; --- 5. Timestamp and schema ---
    result["schema_version"] := "2.0"
    result["timestamp"] := A_Now
    result["privacy_mode"] := includePlaintext ? "plaintext" : "hashed"

    return OkResult(result)
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

SendJsonRpcResult(conn, resultMap, id) {
    ; resultMap is the full handler return Map containing "result" plus optional
    ; perception-action metadata (pre_hash, post_hash, state_changed, outcome_confirmed).
    ; We pass it through as the JSON-RPC "result" field so Python receives all fields.
    response := Map()
    response["jsonrpc"] := "2.0"
    response["result"] := resultMap
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
