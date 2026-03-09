# venv.ps1 — Create a Python virtual environment for MeshXCAD on Windows
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
# 2. Find or install the required Python version
# ============================================================
$Python = $null
$Required = $FreecadPyVer

function Find-PythonVersion {
    param([string]$Ver)

    # Try the Windows Python Launcher (py.exe) first — most reliable on Windows
    try {
        $testVer = & py "-$Ver" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($testVer -eq $Ver) {
            return "py|-$Ver"
        }
    } catch {}

    # Try direct binary names
    foreach ($candidate in @("python$($Ver -replace '\.', '')", "python$Ver", "python")) {
        try {
            $testVer = & $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($testVer -eq $Ver) {
                return $candidate
            }
        } catch {}
    }

    return $null
}

function Invoke-Python {
    # Helper to call python with the right syntax for "py -3.11" vs "python3.11"
    param([string]$PythonCmd, [string[]]$Arguments)
    if ($PythonCmd -match '^py\|(.+)$') {
        $pyArgs = @($Matches[1]) + $Arguments
        & py @pyArgs
    } else {
        & $PythonCmd @Arguments
    }
}

if ($Required) {
    Write-Host ""
    Write-Host "--- Locating Python $Required ---" -ForegroundColor Cyan
    $Python = Find-PythonVersion $Required

    if (-not $Python) {
        Write-Host "Python $Required not found." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "The Windows Python Launcher (py.exe) can manage multiple versions."
        Write-Host "Please install Python $Required from:" -ForegroundColor Yellow
        Write-Host "  https://www.python.org/downloads/" -ForegroundColor White
        Write-Host ""
        Write-Host "During installation, check 'Add Python to PATH' and 'Install py launcher'."
        Write-Host "After installing, re-run this script."
        Write-Host ""
        Write-Host "Alternatively, use the Microsoft Store:"
        Write-Host "  winget install Python.Python.$($Required -replace '\.', '.')"
        Write-Host ""
        Write-Host "Or use pyenv-win:"
        Write-Host "  pyenv install $Required"
        exit 1
    }

    Write-Host "Found Python $Required" -ForegroundColor Green
} else {
    # No FreeCAD version requirement — use system python
    foreach ($candidate in @("python", "python3", "py")) {
        try {
            $ver = & $candidate --version 2>&1
            if ($ver -match "Python 3") {
                $Python = $candidate
                break
            }
        } catch { continue }
    }
}

if (-not $Python) {
    Write-Host "Error: No suitable Python 3 interpreter found." -ForegroundColor Red
    exit 1
}

$PyVersion = Invoke-Python $Python @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
Write-Host "Using Python $PyVersion" -ForegroundColor Cyan

if ($Required -and $PyVersion -ne $Required) {
    Write-Host "ERROR: Interpreter reports $PyVersion but FreeCAD needs $Required." -ForegroundColor Red
    exit 1
}

# ============================================================
# 3. Create the virtual environment
# ============================================================
Write-Host ""
if (Test-Path $VenvDir) {
    Write-Host "Removing existing $VenvDir ..."
    Remove-Item -Recurse -Force $VenvDir
}

Write-Host "Creating virtual environment in $VenvDir (Python $PyVersion) ..."
Invoke-Python $Python @("-m", "venv", $VenvDir)

& "$VenvDir\Scripts\Activate.ps1"

# ============================================================
# 4. Install dependencies
# ============================================================
Write-Host ""
Write-Host "Upgrading pip ..."
pip install --upgrade pip

Write-Host "Installing core dependencies ..."
pip install numpy scipy

Write-Host "Installing visualization dependencies ..."
pip install matplotlib

Write-Host "Installing OpenCASCADE Python bindings ..."
pip install cadquery

Write-Host "Installing test dependencies ..."
pip install pytest

Write-Host "Installing meshxcad in editable mode ..."
pip install -e .

# ============================================================
# 5. Configure FreeCAD PYTHONPATH
# ============================================================
if ($FreecadLib) {
    # Write a .pth file so FreeCAD is importable automatically in the venv
    $sitePackages = python -c "import site; print(site.getsitepackages()[0])"
    $pthFile = Join-Path $sitePackages "freecad_path.pth"
    Set-Content -Path $pthFile -Value $FreecadLib
    Write-Host ""
    Write-Host "Added $FreecadLib to venv site-packages (.pth file)"

    # Verify FreeCAD actually imports
    Write-Host "Verifying FreeCAD import ..."
    try {
        $fcVer = python -c "import FreeCAD; print(f'  FreeCAD {FreeCAD.Version()[0]}.{FreeCAD.Version()[1]} OK')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $fcVer) {
            Write-Host $fcVer -ForegroundColor Green
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
# 6. Summary
# ============================================================
$fcStatus = if ($FreecadLib) { "yes" } else { "no" }
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " MeshXCAD virtual environment ready!" -ForegroundColor Green
Write-Host " Python $PyVersion | FreeCAD: $fcStatus" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Activate it with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Run tests:"
Write-Host "  pytest tests\"
Write-Host ""
Write-Host "CLI usage:"
Write-Host "  python -m meshxcad.cli transfer --plain model.step --detail scan.stl --output result.stl"
Write-Host ""
