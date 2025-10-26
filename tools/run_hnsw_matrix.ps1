param(
    [switch]$UseCTest = $false
)

$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host $msg }
function Run-Cmd([string]$Cmd, [hashtable]$Env = @{}, [string]$Cwd = $null) {
    Write-Host "$ $Cmd"
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $env:ComSpec
    $psi.Arguments = "/c $Cmd"
    if ($Cwd) { $psi.WorkingDirectory = $Cwd }
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    foreach ($k in $Env.Keys) { $psi.Environment[$k] = [string]$Env[$k] }
    $p = [System.Diagnostics.Process]::Start($psi)
    $out = $p.StandardOutput.ReadToEnd() + $p.StandardError.ReadToEnd()
    $p.WaitForExit()
    Write-Host $out
    return @{ Code = $p.ExitCode; Out = $out }
}

$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BUILD = Join-Path $ROOT 'build'
$REL = Join-Path $BUILD 'Release'

function Configure-Build([bool]$Serialize, [bool]$Accelerate) {
    $ser = if ($Serialize) { 'ON' } else { 'OFF' }
    $acc = if ($Accelerate) { 'ON' } else { 'OFF' }
    $args = @(
        'cmake', '-S', '.', '-B', 'build', '-DCMAKE_BUILD_TYPE=Release',
        "-DVESPER_SERIALIZE_BASE_LAYER=$ser",
        "-DVESPER_ENABLE_ACCELERATE=$acc"
    )
    $res = Run-Cmd ($args -join ' ') -Cwd $ROOT
    return ($res.Code -eq 0)
}

function Build-Targets() {
    $res = Run-Cmd 'cmake --build build --config Release --target test_hnsw_batch hnsw_connectivity_test -j' -Cwd $ROOT
    return ($res.Code -eq 0)
}

function Run-Exec([string]$ExeName, [hashtable]$Env) {
    if ($UseCTest) {
        $pattern = [Regex]::Escape([IO.Path]::GetFileNameWithoutExtension($ExeName))
        $res = Run-Cmd "ctest -C Release -R $pattern -V" -Cwd $BUILD -Env $Env
        return @{ Ok = ($res.Code -eq 0); Out = $res.Out }
    } else {
        $exe = Join-Path $REL ($ExeName + '.exe')
        $psiEnv = @{}
        foreach ($k in $Env.Keys) { $psiEnv[$k] = $Env[$k] }
        $res = Run-Cmd ('"' + $exe + '"') -Cwd $BUILD -Env $psiEnv
        return @{ Ok = ($res.Code -eq 0); Out = $res.Out }
    }
}

function Parse-Small([string]$Text) {
    $rate = $null; $cov = $null
    foreach ($line in $Text -split "`r?`n") {
        if ($line -match 'Added in .*\((?<r>[0-9.]+)\s*vec/sec\)') { $rate = [double]$Matches['r'] }
        if ($line -match 'Coverage:\s*(?<c>[0-9.]+)\%') { $cov = [double]$Matches['c'] }
    }
    $ok = ($rate -ne $null -and $cov -ne $null -and $cov -ge 95.0)
    return @{ Ok = $ok; Rate = $rate; Cov = $cov }
}

function Parse-Large([string]$Text) {
    $rate = $null; $cov = $null
    foreach ($line in $Text -split "`r?`n") {
        if ($line -match 'Rate:\s*(?<r>[0-9.]+)\s*vec/sec') { $rate = [double]$Matches['r'] }
        elseif ($line -match 'Build rate:\s*[^0-9]*?(?<r>[0-9.]+)\s*vec/sec') { $rate = [double]$Matches['r'] }
        if ($line -match 'Coverage:\s*(?<c>[0-9.]+)\%') { $cov = [double]$Matches['c'] }
    }
    $ok = ($rate -ne $null -and $cov -ne $null -and $cov -ge 95.0)
    return @{ Ok = $ok; Rate = $rate; Cov = $cov }
}

$rows = @()
$combos = @()
foreach ($serialize in @($false, $true)) { foreach ($accel in @($false, $true)) { $combos += ,@($serialize, $accel) } }
$threadVals = @(2,3,4,6)
$adaptiveVals = @($false,$true)

foreach ($c in $combos) {
    $serialize = $c[0]; $accel = $c[1]
    if (-not (Configure-Build -Serialize:$serialize -Accelerate:$accel)) { Write-Info 'Configure failed'; continue }
    if (-not (Build-Targets)) { Write-Info 'Build failed'; continue }

    foreach ($t in $threadVals) {
        foreach ($ad in $adaptiveVals) {
            $env = @{}
            $env['VESPER_NUM_THREADS'] = [string]$t
            $env['VESPER_EFC'] = '150'
            if ($ad) { $env['VESPER_ADAPTIVE_EF'] = '1' } else { $env.Remove('VESPER_ADAPTIVE_EF') | Out-Null }

            $r1 = Run-Exec -ExeName 'test_hnsw_batch' -Env $env
            $p1 = if ($r1.Ok) { Parse-Small $r1.Out } else { @{ Ok = $false; Rate = $null; Cov = $null } }

            $r2 = Run-Exec -ExeName 'hnsw_connectivity_test' -Env $env
            $p2 = if ($r2.Ok) { Parse-Large $r2.Out } else { @{ Ok = $false; Rate = $null; Cov = $null } }

            $serLabel = if ($serialize) { 'ON' } else { 'OFF' }
            $accLabel = if ($accel) { 'ON' } else { 'OFF' }
            $row = [ordered]@{
                serialize = $serLabel
                accelerate = $accLabel
                threads = $t
                adaptive = $ad
                small_rate = $p1.Rate
                small_cov = $p1.Cov
                large_rate = $p2.Rate
                large_cov = $p2.Cov
                ok = ($p1.Ok -and $p2.Ok)
            }
            $rows += ,$row
        }
    }
}

Write-Host "`n=== HNSW Build Matrix Summary ==="
Write-Host 'serialize accel th adapt | small_rate  small_cov | large_rate  large_cov | ok'
foreach ($r in $rows) {
    $sr = if ($r.small_rate -ne $null) { '{0:N0}' -f $r.small_rate } else { 'NA' }
    $sc = if ($r.small_cov -ne $null) { ('{0:N2}%' -f $r.small_cov) } else { 'NA' }
    $lr = if ($r.large_rate -ne $null) { '{0:N0}' -f $r.large_rate } else { 'NA' }
    $lc = if ($r.large_cov -ne $null) { ('{0:N2}%' -f $r.large_cov) } else { 'NA' }
    $ok = if ($r.ok) { 'PASS' } else { 'FAIL' }
    $line = ('{0,8} {1,5} {2,2} {3,5} | {4,10}  {5,9} | {6,10}  {7,9} | {8}' -f 
        $r.serialize, $r.accelerate, $r.threads, $r.adaptive, $sr, $sc, $lr, $lc, $ok)
    Write-Host $line
}

if ($rows.Count -gt 0 -and ($rows | Where-Object { -not $_.ok } | Measure-Object).Count -eq 0) { exit 0 } else { exit 1 }

