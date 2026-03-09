# venv.ps1 - Create a Python virtual environment for MeshXCAD on Windows
#
# This script detects FreeCAD's required Python version and creates the venv
# with a matching interpreter so that all features work out of the box.
#
# Usage (from PowerShell):
#   .\venv.ps1
#   .\.venv\Scripts\Activate.ps1
#
# If scripts are blocked, run first:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

$ErrorActionPreference = "Stop"

$VenvDir = ".venv"

# ============================================================
# 1. Find FreeCAD and determine its required Python version
# ============================================================
Write-Host "--- Detecting FreeCAD ---" -ForegroundColor Cyan

$FreecadLib = ""
$FreecadPyVer = ""

# Search common Windows install locations for FreeCAD
$FreecadDirs = @()
foreach ($root in @($env:ProgramFiles, ${env:ProgramFiles(x86)}, $env:LOCALAPPDATA)) {
    if (-not $root) { continue }
    $found = Get-ChildItem -Path $root -Filter "FreeCAD*" -Directory -ErrorAction SilentlyContinue
    foreach ($d in $found) {
        if (Test-Path "$($d.FullName)\bin") {
            $FreecadDirs += "$($d.FullName)\bin"
        }
        if (Test-Path "$($d.FullName)\lib") {
            $FreecadDirs += "$($d.FullName)\lib"
        }
    }
}

# Also check if freecad is on PATH
try {
    $fc = Get-Command FreeCADCmd -ErrorAction SilentlyContinue
    if ($fc) {
        $fcDir = Split-Path $fc.Source
        if ($fcDir -notin $FreecadDirs) { $FreecadDirs += $fcDir }
    }
} catch {}
try {
    $fc = Get-Command freecad -ErrorAction SilentlyContinue
    if ($fc) {
        $fcDir = Split-Path $fc.Source
        if ($fcDir -notin $FreecadDirs) { $FreecadDirs += $fcDir }
    }
} catch {}

# Pick the best FreeCAD directory (prefer bin with FreeCAD.pyd)
foreach ($d in $FreecadDirs) {
    if (Test-Path "$d\FreeCAD.pyd") {
        $FreecadLib = $d
        break
    }
    if (Test-Path "$d\FreeCAD.dll") {
        $FreecadLib = $d
        break
    }
}
if (-not $FreecadLib -and $FreecadDirs.Count -gt 0) {
    $FreecadLib = $FreecadDirs[0]
}

# Determine FreeCAD's required Python version by looking for python3XX.dll
if ($FreecadLib) {
    Write-Host "FreeCAD found: $FreecadLib"

    # Look for pythonXY.dll or python3XX.dll in the FreeCAD directory
    $pyDlls = Get-ChildItem -Path $FreecadLib -Filter "python3*.dll" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^python3[0-9]+\.dll$' }

    foreach ($dll in $pyDlls) {
        if ($dll.Name -match '^python(3)([0-9]+)\.dll$') {
            $FreecadPyVer = "$($Matches[1]).$($Matches[2])"
            break
        }
    }

    # Also check parent/sibling directories
    if (-not $FreecadPyVer) {
        $parent = Split-Path $FreecadLib
        $pyDlls = Get-ChildItem -Path $parent -Filter "python3*.dll" -Recurse -Depth 1 -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^python3[0-9]+\.dll$' }
        foreach ($dll in $pyDlls) {
            if ($dll.Name -match '^python(3)([0-9]+)\.dll$') {
                $FreecadPyVer = "$($Matches[1]).$($Matches[2])"
                break
            }
        }
    }
}

if ($FreecadPyVer) {
    Write-Host "FreeCAD requires Python $FreecadPyVer" -ForegroundColor Green
} elseif ($FreecadLib) {
    Write-Host "WARNING: Found FreeCAD but could not determine its Python version." -ForegroundColor Yellow
    Write-Host "         Defaulting to system Python. FreeCAD import may fail."
} else {
    Write-Host "FreeCAD not found." -ForegroundColor Yellow
    Write-Host "FreeCAD features (.FCStd I/O, parametric CAD) will be unavailable."
}

# ============================================================
# 2. Download and install Python 3.11
# ============================================================
$RequiredMajorMinor = if ($FreecadPyVer) { $FreecadPyVer } else { "3.11" }
$PythonInstallDir = Join-Path $PSScriptRoot ".python"

Write-Host ""
Write-Host "--- Installing Python $RequiredMajorMinor ---" -ForegroundColor Cyan

# Find the latest micro release of the required version from python.org
function Get-LatestPythonUrl {
    param([string]$MajorMinor)

    Write-Host "Querying python.org for latest Python $MajorMinor release ..."
    $releasesPage = Invoke-WebRequest -Uri "https://www.python.org/downloads/" -UseBasicParsing
    $pattern = "python-($([regex]::Escape($MajorMinor))\.\d+)-"

    # Find all matching version strings on the downloads page
    $versions = @()
    foreach ($match in [regex]::Matches($releasesPage.Content, $pattern)) {
        $versions += $match.Groups[1].Value
    }
    $versions = $versions | Sort-Object { [version]$_ } -Descending | Select-Object -Unique
    if ($versions.Count -eq 0) {
        Write-Host "ERROR: Could not find Python $MajorMinor on python.org" -ForegroundColor Red
        exit 1
    }
    $latestVer = $versions[0]
    Write-Host "Latest release: Python $latestVer"

    # Build the installer URL (64-bit embeddable or full installer)
    $arch = if ([Environment]::Is64BitOperatingSystem) { "amd64" } else { "win32" }
    $installerUrl = "https://www.python.org/ftp/python/$latestVer/python-$latestVer-$arch.exe"
    return @{ Version = $latestVer; Url = $installerUrl }
}

$pyInfo = Get-LatestPythonUrl $RequiredMajorMinor
$installerPath = Join-Path $env:TEMP "python-$($pyInfo.Version)-installer.exe"

# Download the installer
if (-not (Test-Path $installerPath)) {
    Write-Host "Downloading Python $($pyInfo.Version) ..."
    Write-Host "  $($pyInfo.Url)"
    Invoke-WebRequest -Uri $pyInfo.Url -OutFile $installerPath -UseBasicParsing
    Write-Host "Download complete." -ForegroundColor Green
} else {
    Write-Host "Installer already downloaded: $installerPath"
}

# Install Python to a local directory (no admin required, no PATH pollution)
if (-not (Test-Path (Join-Path $PythonInstallDir "python.exe"))) {
    Write-Host "Installing Python $($pyInfo.Version) to $PythonInstallDir ..."
    $installArgs = @(
        "/quiet"
        "InstallAllUsers=0"
        "TargetDir=$PythonInstallDir"
        "AssociateFiles=0"
        "Shortcuts=0"
        "Include_launcher=0"
        "Include_test=0"
        "Include_pip=1"
        "Include_lib=1"
    )
    Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -NoNewWindow
    if (-not (Test-Path (Join-Path $PythonInstallDir "python.exe"))) {
        Write-Host "ERROR: Python installation failed." -ForegroundColor Red
        Write-Host "Try running: $installerPath"
        exit 1
    }
    Write-Host "Python installed successfully." -ForegroundColor Green
} else {
    Write-Host "Python already installed at $PythonInstallDir"
}

# Use the locally installed Python
$Python = Join-Path $PythonInstallDir "python.exe"

# Verify
$verOutput = & $Python --version 2>&1 | Out-String
Write-Host "Using $($verOutput.Trim())" -ForegroundColor Cyan
if ($verOutput -notmatch "Python $([regex]::Escape($RequiredMajorMinor))") {
    Write-Host "ERROR: Installed Python does not match required $RequiredMajorMinor" -ForegroundColor Red
    exit 1
}

function Invoke-Python {
    param([string]$PythonCmd, [string[]]$Arguments)
    & $PythonCmd @Arguments
}

# ============================================================
# 3. Create the virtual environment
# ============================================================
Write-Host ""
if (Test-Path $VenvDir) {
    Write-Host "Removing existing $VenvDir ..."
    Remove-Item -Recurse -Force $VenvDir
}

Write-Host "Creating virtual environment in $VenvDir ..."
& $Python -m venv $VenvDir

& "$VenvDir\Scripts\Activate.ps1"

# ============================================================
# 4. Configure pip (SSL cert & mirror/trusted-host)
# ============================================================
$PipExtraArgs = @()

# SSL certificate - use SSL_CERT_FILE env var for corporate/custom CAs
if ($env:SSL_CERT_FILE) {
    if (Test-Path $env:SSL_CERT_FILE) {
        Write-Host "Using SSL certificate: $($env:SSL_CERT_FILE)"
        $PipExtraArgs += @("--cert", $env:SSL_CERT_FILE)

        # Write pip.ini so the cert persists inside the venv
        $pipIni = Join-Path $VenvDir "pip.ini"
        $pipIniContent = "[global]" + [Environment]::NewLine + "cert = " + $env:SSL_CERT_FILE
        Set-Content -Path $pipIni -Value $pipIniContent

        # Export so pip's requests library and other tools respect the CA
        $env:PIP_CERT = $env:SSL_CERT_FILE
        $env:REQUESTS_CA_BUNDLE = $env:SSL_CERT_FILE
        $env:CURL_CA_BUNDLE = $env:SSL_CERT_FILE
    } else {
        Write-Host "WARNING: SSL_CERT_FILE is set but file does not exist." -ForegroundColor Yellow
        Write-Host "         Continuing without custom certificate."
    }
}

# Detect pip mirror (Nexus, Artifactory, devpi, etc.) and add --trusted-host
# so that self-signed or corporate-CA HTTPS mirrors work out of the box.
function Get-PipMirrorHosts {
    $indexUrls = @()

    # 1. Check PIP_INDEX_URL environment variable
    if ($env:PIP_INDEX_URL) {
        $indexUrls += $env:PIP_INDEX_URL
    }

    # 2. Check PIP_EXTRA_INDEX_URL environment variable
    if ($env:PIP_EXTRA_INDEX_URL) {
        $indexUrls += $env:PIP_EXTRA_INDEX_URL
    }

    # 3. Check pip config files for index-url / extra-index-url
    if ($indexUrls.Count -eq 0) {
        $pipConfigs = @()
        if ($env:APPDATA) {
            $pipConfigs += Join-Path $env:APPDATA "pip\pip.ini"
            $pipConfigs += Join-Path $env:APPDATA "pip\pip.conf"
        }
        if ($env:USERPROFILE) {
            $pipConfigs += Join-Path $env:USERPROFILE "pip\pip.ini"
            $pipConfigs += Join-Path $env:USERPROFILE ".config\pip\pip.conf"
        }
        foreach ($cfg in $pipConfigs) {
            if ($cfg -and (Test-Path $cfg -ErrorAction SilentlyContinue)) {
                $content = Get-Content $cfg -ErrorAction SilentlyContinue
                foreach ($line in $content) {
                    if ($line -match '^\s*index-url\s*=\s*(.+)$') {
                        $indexUrls += $Matches[1].Trim()
                    }
                    if ($line -match '^\s*extra-index-url\s*=\s*(.+)$') {
                        $indexUrls += $Matches[1].Trim()
                    }
                }
                break
            }
        }
    }

    # Extract hostnames from URLs that are NOT pypi.org
    $hosts = @()
    foreach ($url in $indexUrls) {
        if ($url -match '^https?://([^/:]+)') {
            $h = $Matches[1]
            if ($h -notlike '*pypi.org' -and $h -notlike '*pythonhosted.org') {
                $hosts += $h
            }
        }
    }

    return ($hosts | Select-Object -Unique)
}

$MirrorHosts = Get-PipMirrorHosts
if ($MirrorHosts) {
    Write-Host "Detected pip mirror host(s) - adding --trusted-host:" -ForegroundColor Cyan
    foreach ($h in $MirrorHosts) {
        Write-Host "  $h"
        $PipExtraArgs += @("--trusted-host", $h)
    }
}

# ============================================================
# 5. Install dependencies
# ============================================================
Write-Host ""
Write-Host "Upgrading pip ..."
pip install --upgrade pip @PipExtraArgs

Write-Host "Installing core dependencies ..."
pip install numpy scipy @PipExtraArgs

Write-Host "Installing visualization dependencies ..."
pip install matplotlib @PipExtraArgs

Write-Host "Installing OpenCASCADE Python bindings ..."
pip install cadquery @PipExtraArgs

Write-Host "Installing test dependencies ..."
pip install pytest @PipExtraArgs

Write-Host "Installing meshxcad in editable mode ..."
pip install -e . @PipExtraArgs

# ============================================================
# 6. Configure FreeCAD PYTHONPATH
# ============================================================
if ($FreecadLib) {
    # Write a .pth file so FreeCAD is importable automatically in the venv
    $siteCmd = 'import site; print(site.getsitepackages()[0])'
    $sitePackages = python -c $siteCmd
    $pthFile = Join-Path $sitePackages "freecad_path.pth"
    Set-Content -Path $pthFile -Value $FreecadLib
    Write-Host ""
    Write-Host "Added $FreecadLib to venv site-packages (.pth file)"

    # Verify FreeCAD actually imports
    Write-Host "Verifying FreeCAD import ..."
    try {
        $verifyCmd = 'import FreeCAD; v=FreeCAD.Version(); print("FreeCAD " + v[0] + "." + v[1] + " OK")'
        $fcVer = python -c $verifyCmd 2>$null
        if ($LASTEXITCODE -eq 0 -and $fcVer) {
            Write-Host "  $fcVer" -ForegroundColor Green
            Write-Host "  FreeCAD is working." -ForegroundColor Green
        } else {
            Write-Host "  WARNING: FreeCAD import failed." -ForegroundColor Yellow
            Write-Host "  The library may have additional dependencies."
            Write-Host "  Check: python -c 'import FreeCAD'"
        }
    } catch {
        Write-Host "  WARNING: FreeCAD import failed." -ForegroundColor Yellow
        Write-Host "  Check: python -c 'import FreeCAD'"
    }
}

# ============================================================
# 7. Summary
# ============================================================
$fcStatus = if ($FreecadLib) { "yes" } else { "no" }
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " MeshXCAD virtual environment ready!" -ForegroundColor Green
Write-Host " Python $RequiredMajorMinor | FreeCAD: $fcStatus" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Activate it with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Run tests:"
Write-Host "  pytest tests/"
Write-Host ""
Write-Host "CLI usage:"
Write-Host "  python -m meshxcad.cli transfer --plain model.step --detail scan.stl --output result.stl"
Write-Host ""
