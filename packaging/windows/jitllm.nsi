; jitllm's Windows setup: a per-user install, no administrator.
;
; Built by packaging/windows/build.sh with makensis (NSIS 3.08 or later),
; which runs on the Linux release runner. Defines it passes:
;
;   VERSION   the release version, 1.2.3 or 1.2.3-rc1
;   VERSION4  the version as four numbers, a suffix dropped, for the file version resource
;   ARCH      amd64 or arm64; the binaries are that architecture's, the
;             installer itself is x86 code, which arm64 Windows runs
;   SRC       a directory holding jitllm.exe, jitllmd.exe, jitllm-desktop.exe,
;             jitllm.ico and LICENSE.txt
;   OUTFILE   the setup program to write
;   SIGN      optional: a command signing "%1", run on the uninstaller and
;             on the setup program once each is written (!uninstfinalize,
;             !finalize)
;
; What it does, all under the person's own account:
;   - %LOCALAPPDATA%\Programs\jitllm holds the three programs (the directory
;     scripts/install.ps1 uses too, so either replaces the other's copy);
;   - that directory goes on the user PATH (optional, on by default);
;   - Start menu shortcuts for the desktop app and the uninstaller;
;   - optionally, jitllmd starts at login on 127.0.0.1:8080, over the desktop
;     app's model folder, with no console window;
;   - an Add/Remove Programs entry whose uninstaller undoes each of these.
;
; /S installs silently with the defaults (winget passes it); /D=<dir> as the
; last argument picks another directory.

Unicode true
ManifestDPIAware true
RequestExecutionLevel user
SetCompressor /SOLID lzma

!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "LogicLib.nsh"
!include "Sections.nsh"

!define NAME "jitllm"
!define PUBLISHER "jitllm"
!define URL "https://jitllm.org"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\jitllm"
!define RUN_KEY "Software\Microsoft\Windows\CurrentVersion\Run"
!define LOGIN_SCRIPT "jitllmd-login.vbs"

Name "${NAME}"
OutFile "${OUTFILE}"
InstallDir "$LOCALAPPDATA\Programs\jitllm"
InstallDirRegKey HKCU "Software\jitllm" "InstallDir"
BrandingText "${NAME} ${VERSION} (${ARCH})"

VIProductVersion "${VERSION4}"
VIAddVersionKey "ProductName" "${NAME}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "FileDescription" "${NAME} setup"
VIAddVersionKey "CompanyName" "${PUBLISHER}"
VIAddVersionKey "LegalCopyright" "Apache-2.0"

!ifdef SIGN
!uninstfinalize '${SIGN}'
!finalize '${SIGN}'
!endif

!define MUI_ICON "${SRC}/jitllm.ico"
!define MUI_UNICON "${SRC}/jitllm.ico"
!define MUI_ABORTWARNING
!define MUI_COMPONENTSPAGE_NODESC
!define MUI_FINISHPAGE_RUN "$INSTDIR\jitllm-desktop.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Open jitllm"

!insertmacro MUI_PAGE_LICENSE "${SRC}/LICENSE.txt"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

; stopRunning ends this install's own jitllmd and desktop app, which hold
; their .exe files open; a jitllm installed elsewhere is left running.
!macro stopRunning
  nsExec::Exec `powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "Get-Process jitllmd,jitllm-desktop,jitllm -ErrorAction SilentlyContinue | Where-Object { $$_.Path -like '$INSTDIR\*' } | Stop-Process -Force"`
  Pop $0
!macroend

Section "jitllm (required)" SecMain
  SectionIn RO
  !insertmacro stopRunning
  SetOutPath "$INSTDIR"
  File "${SRC}/jitllm.exe"
  File "${SRC}/jitllmd.exe"
  File "${SRC}/jitllm-desktop.exe"
  File "${SRC}/jitllm.ico"
  File "${SRC}/LICENSE.txt"
  WriteUninstaller "$INSTDIR\uninstall.exe"
  WriteRegStr HKCU "Software\jitllm" "InstallDir" "$INSTDIR"

  CreateDirectory "$SMPROGRAMS\jitllm"
  CreateShortcut "$SMPROGRAMS\jitllm\jitllm.lnk" "$INSTDIR\jitllm-desktop.exe" "" "$INSTDIR\jitllm-desktop.exe" 0
  CreateShortcut "$SMPROGRAMS\jitllm\Uninstall jitllm.lnk" "$INSTDIR\uninstall.exe"

  WriteRegStr HKCU "${UNINST_KEY}" "DisplayName" "${NAME}"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINST_KEY}" "Publisher" "${PUBLISHER}"
  WriteRegStr HKCU "${UNINST_KEY}" "URLInfoAbout" "${URL}"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\jitllm.ico"
  WriteRegStr HKCU "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINST_KEY}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegStr HKCU "${UNINST_KEY}" "QuietUninstallString" '"$INSTDIR\uninstall.exe" /S'
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoRepair" 1
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKCU "${UNINST_KEY}" "EstimatedSize" "$0"
SectionEnd

; The user PATH, through .NET's Environment, which also tells running
; programs (Explorer, a new terminal) that it changed. A directory already
; there is not added twice.
Section "Add jitllm to PATH" SecPath
  nsExec::ExecToLog `powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$$d = '$INSTDIR'; $$p = [Environment]::GetEnvironmentVariable('Path', 'User'); if ((($$p -split ';') -notcontains $$d)) { [Environment]::SetEnvironmentVariable('Path', ($$(if ($$p) { $$p + ';' }) + $$d), 'User') }"`
  Pop $0
  WriteRegDWORD HKCU "Software\jitllm" "AddedToPath" 1
SectionEnd

; jitllmd is a console program; started from the Run key it would open a
; console window at every login. The script starts it hidden (window style 0)
; and does not wait for it.
Section /o "Start the server at login" SecLogin
  FileOpen $0 "$INSTDIR\${LOGIN_SCRIPT}" w
  FileWrite $0 `CreateObject("WScript.Shell").Run """$INSTDIR\jitllmd.exe"" serve -addr 127.0.0.1:8080 -models ""$LOCALAPPDATA\jitllm\models""", 0, False$\r$\n`
  FileClose $0
  CreateDirectory "$LOCALAPPDATA\jitllm\models"
  WriteRegStr HKCU "${RUN_KEY}" "jitllmd" '"$SYSDIR\wscript.exe" "$INSTDIR\${LOGIN_SCRIPT}"'
  ; And now, so the server is up without logging out and in.
  Exec '"$SYSDIR\wscript.exe" "$INSTDIR\${LOGIN_SCRIPT}"'
SectionEnd

; An upgrade that leaves "start at login" unticked removes an earlier
; install's login entry, so the ticked boxes are the state after setup.
Section "-login off"
  ${IfNot} ${SectionIsSelected} ${SecLogin}
    DeleteRegValue HKCU "${RUN_KEY}" "jitllmd"
    Delete "$INSTDIR\${LOGIN_SCRIPT}"
  ${EndIf}
SectionEnd

Section "Uninstall"
  !insertmacro stopRunning
  DeleteRegValue HKCU "${RUN_KEY}" "jitllmd"
  ReadRegDWORD $1 HKCU "Software\jitllm" "AddedToPath"
  ${If} $1 == 1
    nsExec::ExecToLog `powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$$d = '$INSTDIR'; $$p = [Environment]::GetEnvironmentVariable('Path', 'User'); [Environment]::SetEnvironmentVariable('Path', ((($$p -split ';') | Where-Object { $$_ -and $$_ -ne $$d }) -join ';'), 'User')"`
    Pop $0
  ${EndIf}
  Delete "$INSTDIR\jitllm.exe"
  Delete "$INSTDIR\jitllmd.exe"
  Delete "$INSTDIR\jitllm-desktop.exe"
  Delete "$INSTDIR\jitllm.ico"
  Delete "$INSTDIR\LICENSE.txt"
  Delete "$INSTDIR\${LOGIN_SCRIPT}"
  Delete "$INSTDIR\uninstall.exe"
  ; Only if empty: a file the person put there stays, and so does its folder.
  RMDir "$INSTDIR"
  Delete "$SMPROGRAMS\jitllm\jitllm.lnk"
  Delete "$SMPROGRAMS\jitllm\Uninstall jitllm.lnk"
  RMDir "$SMPROGRAMS\jitllm"
  DeleteRegKey HKCU "${UNINST_KEY}"
  DeleteRegKey HKCU "Software\jitllm"
  ; Models, chats and settings, in the desktop app's data folders, are
  ; the person's data and are kept.
SectionEnd
