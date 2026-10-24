<#
.SYNOPSIS
  Downloads the Intel oneMKL runtime that MatrixEigenValueAndEigenVector_MKL.cpp needs into mkl\bin next to this script.

.DESCRIPTION
  Fetches Intel's official "mkl" and "intel-openmp" wheels for Windows x64 from PyPI, checks each file against the
  SHA-256 digest that PyPI publishes (for the default version also against digests pinned in this script), and
  extracts only what the drop-in needs:

    mkl\bin\       mkl_rt, mkl_core, mkl_intel_thread, mkl_def, mkl_mc3, mkl_avx2, mkl_avx512, mkl_avx10 (2026 and
                   later) and libiomp5md
    mkl\licenses\  the oneMKL and Intel OpenMP license texts (ship them with the DLLs)

  Nothing is installed system-wide, no administrator rights are needed and Python is not required.
  The import library mkl\lib\mkl_rt.lib is already in the repository.

  oneMKL 2024.2.2 and 2025.0.0 are refused: on AMD processors under Windows their dgeev writes past its workspace
  (memory corruption) and can hang. 2025.0.1 and later are fixed.

.PARAMETER Version
  oneMKL version to fetch. Default 2026.2.0, the version the drop-in was tested with. Must be 2025.0.1 or newer.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File install_mkl.ps1
  powershell -ExecutionPolicy Bypass -File install_mkl.ps1 -Version 2025.3.1
#>
param([string]$Version = "2026.2.0")

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # the progress bar makes Invoke-WebRequest very slow in Windows PowerShell 5.1
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

# SHA-256 of the default version's wheels, pinned here so that the default install does not rely on PyPI alone.
# Other versions are checked against the digests that PyPI publishes.
$Pinned = @{
    "mkl-2026.2.0-py2.py3-none-win_amd64.whl"          = "5f2c187f0dda47efa19c1922cb3c8815924b8e5ed1613d28a3a69ec2327d4045"
    "intel_openmp-2026.2.0-py2.py3-none-win_amd64.whl" = "0fcd60701d6dcf8f17cd40c9733d7046ff3b82921a0a039bea2e6a68fb12c152"
}

if ([version]$Version -lt [version]"2025.0.1") {
    throw "oneMKL $Version is not supported: 2024.2.2 and 2025.0.0 have a dgeev bug on AMD/Windows (out-of-bounds writes). Use 2025.0.1 or newer."
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$bin = Join-Path $root "mkl\bin"
$licenses = Join-Path $root "mkl\licenses"
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("onemkl-" + [guid]::NewGuid().ToString("N"))
# Everything is extracted into a staging folder first; mkl\bin is replaced only after all files are there, so a failed
# download leaves a previous install untouched.
$stageBin = Join-Path $tmp "bin"
$stageLicenses = Join-Path $tmp "licenses"
New-Item -ItemType Directory -Path $tmp, $stageBin, $stageLicenses | Out-Null
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Get-Wheel([string]$package) {
    $meta = Invoke-RestMethod -UseBasicParsing "https://pypi.org/pypi/$package/$Version/json"
    $file = $meta.urls | Where-Object { $_.filename -like "*-win_amd64.whl" } | Select-Object -First 1
    if (-not $file) { throw "PyPI has no Windows x64 wheel for $package $Version" }
    $path = Join-Path $tmp $file.filename
    Write-Host ("Downloading {0} ({1} MB)" -f $file.filename, [math]::Round($file.size / 1MB))
    Invoke-WebRequest -UseBasicParsing -Uri $file.url -OutFile $path
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
    if ($hash -ne $file.digests.sha256) { throw "SHA-256 mismatch for $($file.filename): got $hash" }
    if ($Pinned.ContainsKey($file.filename) -and $hash -ne $Pinned[$file.filename]) {
        throw "SHA-256 of $($file.filename) differs from the digest pinned in this script: got $hash"
    }
    return $path
}

# Extracts the wheel entries whose path matches one of $patterns (wildcards) into $dest; $rename maps a file name
# to the name it is written under.
function Expand-Entries([string]$wheel, [string[]]$patterns, [string]$dest, [hashtable]$rename = @{}) {
    $zip = [IO.Compression.ZipFile]::OpenRead($wheel)
    try {
        foreach ($entry in $zip.Entries) {
            foreach ($pattern in $patterns) {
                if ($entry.FullName -like $pattern) {
                    $name = if ($rename.ContainsKey($entry.Name)) { $rename[$entry.Name] } else { $entry.Name }
                    [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $dest $name), $true)
                    break
                }
            }
        }
    } finally {
        $zip.Dispose()
    }
}

try {
    $mkl = Get-Wheel "mkl"
    $dlls = "mkl_rt", "mkl_core", "mkl_intel_thread", "mkl_def", "mkl_mc3", "mkl_avx2", "mkl_avx512", "mkl_avx10" |
        ForEach-Object { "*/Library/bin/$_.*.dll" }
    Expand-Entries $mkl $dlls $stageBin
    Expand-Entries $mkl @("*.dist-info/LICENSE.txt") $stageLicenses @{ "LICENSE.txt" = "oneMKL-LICENSE.txt" }

    $omp = Get-Wheel "intel-openmp"
    Expand-Entries $omp @("*/Library/bin/libiomp5md.dll") $stageBin
    Expand-Entries $omp @("*.dist-info/LICENSE.txt", "*/third-party-programs.txt") $stageLicenses @{
        "LICENSE.txt" = "intel-openmp-LICENSE.txt"; "third-party-programs.txt" = "intel-openmp-third-party-programs.txt" }

    foreach ($required in "mkl_rt.*.dll", "mkl_core.*.dll", "mkl_intel_thread.*.dll", "mkl_def.*.dll", "libiomp5md.dll") {
        if (-not (Get-ChildItem -LiteralPath $stageBin -Filter $required)) { throw "$required was not found in the wheels" }
    }
    # Replace the previous install as a whole, so that DLLs of two versions are never mixed.
    New-Item -ItemType Directory -Force -Path $bin, $licenses | Out-Null
    Get-ChildItem -LiteralPath $bin -Filter *.dll | Remove-Item -Force
    Get-ChildItem -LiteralPath $stageBin | Move-Item -Destination $bin -Force
    Get-ChildItem -LiteralPath $stageLicenses | Move-Item -Destination $licenses -Force
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
$files = Get-ChildItem -LiteralPath $bin -Filter *.dll
Write-Host ("Installed oneMKL {0}: {1} DLLs, {2} MB, in {3}" -f $Version, $files.Count,
    [math]::Round(($files | Measure-Object Length -Sum).Sum / 1MB), $bin)
$files | ForEach-Object { Write-Host ("  {0,-24} {1,5} MB" -f $_.Name, [math]::Round($_.Length / 1MB)) }

if (-not (Get-ChildItem -LiteralPath $bin -Filter "mkl_rt.3.dll")) {
    Write-Warning ("This oneMKL names its runtime differently from mkl_rt.3.dll, which mkl\lib\mkl_rt.lib links to. " +
        "Change the first line of mkl\lib\mkl_rt.def to the name of the mkl_rt DLL above and regenerate the import " +
        "library in a Visual Studio developer prompt: lib /def:mkl_rt.def /machine:x64 /out:mkl_rt.lib")
}
Write-Host "Copy the DLLs in mkl\bin next to the .exe that loads your DLL, or add mkl\bin to PATH."
