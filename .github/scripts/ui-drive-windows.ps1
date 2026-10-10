# Drive a running jitllm-ui window on Windows with synthetic input: a resize
# storm, minimize and restore, clicks on the sidebar, typing a message and
# sending it, Escape to stop the reply. The window is found by process id;
# positions are fractions of its client area, so they follow the window
# wherever Windows put it. Exits non-zero if the process is gone after any
# step.
#
#     ui-drive-windows.ps1 -ProcessId <pid> -Shot <png>
param([int]$ProcessId, [string]$Shot)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class W {
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref System.Drawing.Point p);
  [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr h, int x, int y, int w, int hh, bool repaint);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(int f, int x, int y, int d, int e);
}
'@ -ReferencedAssemblies System.Drawing

function Alive($step) {
  if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
    Write-Host "::error::jitllm-ui died during: $step"
    exit 1
  }
  Write-Host "ok: $step"
}

$p = Get-Process -Id $ProcessId
for ($i = 0; $i -lt 50 -and $p.MainWindowHandle -eq 0; $i++) { Start-Sleep -Milliseconds 200; $p.Refresh() }
$h = $p.MainWindowHandle
if ($h -eq 0) { Write-Host "::error::jitllm-ui has no window"; exit 1 }

function Click([double]$fx, [double]$fy) {
  $r = New-Object W+RECT
  [void][W]::GetClientRect($h, [ref]$r)
  $pt = New-Object System.Drawing.Point ([int]($r.R * $fx)), ([int]($r.B * $fy))
  [void][W]::ClientToScreen($h, [ref]$pt)
  [void][W]::SetCursorPos($pt.X, $pt.Y)
  [W]::mouse_event(0x2, 0, 0, 0, 0); [W]::mouse_event(0x4, 0, 0, 0, 0)
  Start-Sleep -Milliseconds 300
}

[void][W]::SetForegroundWindow($h)
for ($i = 0; $i -lt 30; $i++) {
  [void][W]::MoveWindow($h, 20, 20, 960 + ($i * 37) % 500, 640 + ($i * 53) % 300, $true)
  Start-Sleep -Milliseconds 50
}
[void][W]::MoveWindow($h, 20, 20, 1200, 800, $true)
Alive 'resize storm'
for ($i = 0; $i -lt 3; $i++) {
  [void][W]::ShowWindow($h, 6); Start-Sleep -Milliseconds 400   # SW_MINIMIZE
  [void][W]::ShowWindow($h, 9); Start-Sleep -Milliseconds 400   # SW_RESTORE
}
Alive 'minimize and restore'
[void][W]::SetForegroundWindow($h)
# The sidebar: Models, Machine, Discover, then back to Chat.
foreach ($y in 0.20, 0.30, 0.155, 0.107) { Click 0.06 $y }
Alive 'tab clicks'
# The composer, a message, Enter, then Escape mid-reply; twice.
for ($i = 0; $i -lt 2; $i++) {
  Click 0.40 0.87
  [System.Windows.Forms.SendKeys]::SendWait("Once upon a time{ENTER}")
  Start-Sleep -Milliseconds 700
  [System.Windows.Forms.SendKeys]::SendWait("{ESC}")
  Start-Sleep -Milliseconds 500
}
Alive 'send and stop'
# The theme toggle at the foot of the sidebar, there and back.
Click 0.06 0.925; Click 0.06 0.925
Alive 'theme swap'
Start-Sleep -Seconds 2
$b = [System.Windows.Forms.SystemInformation]::VirtualScreen
$bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($b.Left, $b.Top, 0, 0, $bmp.Size)
$bmp.Save($Shot)
Alive 'end'
