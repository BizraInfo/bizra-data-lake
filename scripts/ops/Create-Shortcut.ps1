$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\BIZRA Data Lake.lnk")
$Shortcut.TargetPath = "C:\BIZRA-DATA-LAKE\scripts\ops\CONTROL-CENTER.bat"
$Shortcut.WorkingDirectory = "C:\BIZRA-DATA-LAKE\scripts\ops"
$Shortcut.IconLocation = "C:\Windows\System32\imageres.dll,185"
$Shortcut.Description = "BIZRA Data Lake Control Center - Single Source of Truth"
$Shortcut.Save()

Write-Host "✅ Desktop shortcut created successfully!" -ForegroundColor Green
