# venv.ps1 - Create a Python virtual environment for MeshXCAD on Windows
#
# This script detects FreeCAD's required Python version and creates the venv
# with a matching interpreter so that all features work out of the box.
#
# Usage (from PowerShell):
#   .\venv.ps1
#   .\.venv\Scripts\Activate.ps1
#
# To point at a specific Python install, set one of:
#   $env:MESHXCAD_PYTHON = "C:\Python311\python.exe"
#   $env:MESHXCAD_PYTHON_DIR = "C:\Python311"
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
# 2. Locate Python
# ============================================================
#
# Resolution order:
#   1. MESHXCAD_PYTHON env var  - full path to python.exe
#   2. MESHXCAD_PYTHON_DIR env var - directory containing python.exe
#   3. Search PATH for python / python3 / py matching required version
#   4. Fallback: any Python 3 on PATH
#
$RequiredVer = if ($FreecadPyVer) { $FreecadPyVer } else { "3.11" }
$Python = $null

Write-Host ""
Write-Host "--- Locating Python $RequiredVer ---" -ForegroundColor Cyan

# Helper: run a candidate and check --version output
function Test-PythonExe {
    param([string]$Exe, [string]$Ver)
    $ErrorActionPreference = "Continue"
    try {
        $out = & $Exe --version 2>&1 | Out-String
        if ($out -match "Python (\d+\.\d+)") {
            if ($Matches[1] -eq $Ver) { return $true }
        }
    } catch {}
    return $false
}

# 1. MESHXCAD_PYTHON - explicit path to python.exe
if ($env:MESHXCAD_PYTHON) {
    Write-Host "MESHXCAD_PYTHON is set: $($env:MESHXCAD_PYTHON)"
    if (Test-Path $env:MESHXCAD_PYTHON) {
        if (Test-PythonExe $env:MESHXCAD_PYTHON $RequiredVer) {
            $Python = $env:MESHXCAD_PYTHON
            Write-Host "  Matched Python $RequiredVer" -ForegroundColor Green
        } else {
            Write-Host "  WARNING: exists but is not Python $RequiredVer" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  WARNING: file not found" -ForegroundColor Yellow
    }
}

# 2. MESHXCAD_PYTHON_DIR - directory containing python.exe
if (-not $Python -and $env:MESHXCAD_PYTHON_DIR) {
    Write-Host "MESHXCAD_PYTHON_DIR is set: $($env:MESHXCAD_PYTHON_DIR)"
    $candidate = Join-Path $env:MESHXCAD_PYTHON_DIR "python.exe"
    if (Test-Path $candidate) {
        if (Test-PythonExe $candidate $RequiredVer) {
            $Python = $candidate
            Write-Host "  Matched Python $RequiredVer" -ForegroundColor Green
        } else {
            Write-Host "  WARNING: python.exe found but is not Python $RequiredVer" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  WARNING: python.exe not found in directory" -ForegroundColor Yellow
    }
}

# 3. Search PATH for matching version
if (-not $Python) {
    $ErrorActionPreference = "Continue"
    foreach ($candidate in @("python", "python3", "py")) {
        try {
            $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
            if ($cmd) {
                if (Test-PythonExe $cmd.Source $RequiredVer) {
                    $Python = $cmd.Source
                    Write-Host "Found on PATH: $Python" -ForegroundColor Green
                    break
                }
            }
        } catch {}
    }
    $ErrorActionPreference = "Stop"
}

# 4. Fallback: any Python 3 on PATH (if no specific version required by FreeCAD)
if (-not $Python -and -not $FreecadPyVer) {
    $ErrorActionPreference = "Continue"
    foreach ($candidate in @("python", "python3", "py")) {
        try {
            $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
            if ($cmd) {
                $out = & $cmd.Source --version 2>&1 | Out-String
                if ($out -match "Python 3\.") {
                    $Python = $cmd.Source
                    Write-Host "Fallback: using $Python" -ForegroundColor Yellow
                    break
                }
            }
        } catch {}
    }
    $ErrorActionPreference = "Stop"
}

if (-not $Python) {
    Write-Host ""
    Write-Host "ERROR: Python $RequiredVer not found." -ForegroundColor Red
    Write-Host ""
    Write-Host "Set one of these environment variables before running this script:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host '  $env:MESHXCAD_PYTHON = "C:\path\to\python.exe"'
    Write-Host '  $env:MESHXCAD_PYTHON_DIR = "C:\path\to\python\directory"'
    Write-Host ""
    Write-Host "Example:"
    Write-Host '  $env:MESHXCAD_PYTHON = "C:\Python311\python.exe"'
    Write-Host "  .\venv.ps1"
    Write-Host ""
    exit 1
}

# Show which Python we are using
$verOutput = & $Python --version 2>&1 | Out-String
Write-Host "Using $($verOutput.Trim()) at $Python" -ForegroundColor Cyan

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
Write-Host " Python $RequiredVer | FreeCAD: $fcStatus" -ForegroundColor Green
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
