# Installs jitllm from a GitHub release on Windows.
#
#   irm https://jitllm.org/install.ps1 | iex
#
# $env:JITLLM_PROGRAMS names the programs to install, separated by spaces or
# commas: jitllm (the CLI), jitllmd (the server), desktop (jitllm-desktop),
# tui (jitllm-tui), or all. Unset, it installs jitllm and jitllmd. Run as a
# file, the same is -Programs. Each archive is checked against the release's
# checksums.txt before anything is installed, and the install directory is
# added to the user's PATH.
#
#   $env:JITLLM_VERSION      a release tag, such as v0.1.0 (default: the latest)
#   $env:JITLLM_INSTALL_DIR  where the binaries go
#                            (default: %LOCALAPPDATA%\Programs\jitllm)
#   $env:JITLLM_DOWNLOAD_URL where releases are fetched from (default: GitHub)
param(
    [string[]]$Programs = @(),
    [string]$Version = $env:JITLLM_VERSION,
    [string]$InstallDir = $env:JITLLM_INSTALL_DIR
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$repo = 'jitllm/jitllm'
$base = if ($env:JITLLM_DOWNLOAD_URL) { $env:JITLLM_DOWNLOAD_URL } else { "https://github.com/$repo/releases/download" }

if (-not $Programs -and $env:JITLLM_PROGRAMS) { $Programs = $env:JITLLM_PROGRAMS -split '[\s,]+' | Where-Object { $_ } }
if (-not $Programs) { $Programs = @('jitllm', 'jitllmd') }
$progs = foreach ($p in $Programs) {
    switch ($p) {
        'all' { 'jitllm', 'jitllmd', 'jitllm-desktop', 'jitllm-tui' }
        { $_ -in 'jitllm', 'cli' } { 'jitllm' }
        { $_ -in 'jitllmd', 'server' } { 'jitllmd' }
        { $_ -in 'desktop', 'jitllm-desktop', 'ui' } { 'jitllm-desktop' }
        { $_ -in 'tui', 'jitllm-tui' } { 'jitllm-tui' }
        default { throw "jitllm install: unknown program $p (jitllm, jitllmd, desktop, tui, all)" }
    }
}
$progs = $progs | Select-Object -Unique

$arch = switch ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture) {
    'X64' { 'amd64' }
    'Arm64' { 'arm64' }
    default { throw "jitllm install: $_ is not a release architecture (amd64, arm64)" }
}

if (-not $Version) {
    # The latest release's page redirects to its tag; no API call, no token.
    try { $r = Invoke-WebRequest -UseBasicParsing -Method Head -Uri "https://github.com/$repo/releases/latest" }
    catch { throw "jitllm install: no release found at https://github.com/$repo/releases" }
    $final = if ($r.BaseResponse.ResponseUri) { $r.BaseResponse.ResponseUri.AbsoluteUri } else { $r.BaseResponse.RequestMessage.RequestUri.AbsoluteUri }
    $Version = ($final -split '/')[-1]
    if ($Version -notlike 'v*') { throw "jitllm install: no release found at https://github.com/$repo/releases" }
}
$ver = $Version.TrimStart('v')
if (-not $InstallDir) { $InstallDir = Join-Path $env:LOCALAPPDATA 'Programs\jitllm' }

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("jitllm-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    Write-Host "jitllm $Version for windows/$arch"
    $sums = Join-Path $tmp 'checksums.txt'
    Invoke-WebRequest -UseBasicParsing -Uri "$base/$Version/checksums.txt" -OutFile $sums
    $want = @{}
    foreach ($line in Get-Content $sums) {
        $f = $line -split '\s+'
        if ($f.Count -ge 2) { $want[$f[1]] = $f[0] }
    }
    foreach ($p in $progs) {
        $archive = "${p}_${ver}_windows_${arch}.zip"
        $file = Join-Path $tmp $archive
        Invoke-WebRequest -UseBasicParsing -Uri "$base/$Version/$archive" -OutFile $file
        if (-not $want.ContainsKey($archive)) { throw "jitllm install: $archive is not in checksums.txt" }
        if ((Get-FileHash -Algorithm SHA256 $file).Hash -ne $want[$archive].ToUpper()) { throw "jitllm install: $archive does not match its checksum" }
        Expand-Archive -Path $file -DestinationPath (Join-Path $tmp $p)
    }
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    foreach ($p in $progs) {
        Copy-Item -Force (Join-Path $tmp "$p\$p.exe") (Join-Path $InstallDir "$p.exe")
        Write-Host "installed $(Join-Path $InstallDir "$p.exe")"
    }
} finally {
    Remove-Item -Recurse -Force $tmp
}

if ($env:OS -eq 'Windows_NT') {
    $path = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (($path -split ';') -notcontains $InstallDir) {
        [Environment]::SetEnvironmentVariable('Path', ($(if ($path) { "$path;" }) + $InstallDir), 'User')
        Write-Host "added $InstallDir to your PATH; open a new terminal to use it"
    }
}
