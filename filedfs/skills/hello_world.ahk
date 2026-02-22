; BIZRA Skill: hello_world
; ============================================================
; Smoke-test skill — reads inputs from temp JSON file,
; returns a greeting. Used to verify the skill dispatch pipeline.
;
; Usage: ahk_bridge.ahk calls this with:
;   invoke_skill({ skill: "hello_world", inputs: { name: "BIZRA" } })
;
; Input file (argv[1]): JSON with optional "name" field
; Output: stdout line consumed by bridge
; ============================================================

#Requires AutoHotkey v2.0

; Read input file path from command line
inputFile := A_Args.Length > 0 ? A_Args[1] : ""

name := "World"
if (inputFile != "" && FileExist(inputFile)) {
    try {
        content := FileRead(inputFile, "UTF-8")
        inputs := Jxon_Load(&content)
        if inputs.Has("name")
            name := inputs["name"]
    }
}

; Output result to stdout (bridge captures this)
FileAppend("Hello, " name "! BIZRA seed is alive.`n", "*", "UTF-8")
ExitApp 0
