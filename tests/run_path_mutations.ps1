param(
  [string]$MujocoRoot = 'C:/Users/lethu/Downloads/mujoco-3.16.0-windows-x86_64',
  [string]$Compiler = 'C:/msys64/ucrt64/bin/g++.exe'
)
$ErrorActionPreference = 'Stop'
$project = Split-Path $PSScriptRoot -Parent
$output = Join-Path $project 'build/batch002-mutations'
New-Item -ItemType Directory -Force $output | Out-Null
$original = Join-Path $project 'src/feasibility.cpp'
$hash = (Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash
$source = Get-Content -LiteralPath $original -Raw
$copy = Join-Path $output 'mutant.cpp'
$mutations = @(
  @('skip-final-waypoint', 'waypoint_index < waypoints.size()', 'waypoint_index + 1 < waypoints.size()'),
  @('skip-final-joint', 'joint_index < joint_count', 'joint_index + 1 < joint_count'),
  @('reject-equal-step', 'std::abs(angle - previous_angle) > max_joint_step_rad', 'std::abs(angle - previous_angle) >= max_joint_step_rad'),
  @('admit-equal-bounds', 'limit.lower >= limit.upper', 'limit.lower > limit.upper'),
  @('admit-nonfinite-angle', 'if (!std::isfinite(angle))', 'if (false)'),
  @('admit-extra-joints', 'waypoint.size() != joint_count', 'waypoint.size() < joint_count')
)
Set-Content -LiteralPath $copy -Value $source -NoNewline
& cmake -S (Join-Path $PSScriptRoot 'standalone') -B $output -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_COMPILER=$Compiler" "-DMUJOCO_ROOT=$MujocoRoot" "-DRELIC_FEASIBILITY_SOURCE=$copy"
if ($LASTEXITCODE -ne 0) { throw 'mutation configure failed' }
foreach ($mutation in $mutations) {
  $label, $find, $replace = $mutation
  if (-not $source.Contains($find)) { throw "mutation pattern missing: $label" }
  Set-Content -LiteralPath $copy -Value $source.Replace($find, $replace) -NoNewline
  & cmake --build $output -j 2
  if ($LASTEXITCODE -ne 0) { throw "mutation build failed: $label" }
  Push-Location $output
  try {
    & ./relic_path_standalone.exe fuzz 0x52454c494332 1000
    if ($LASTEXITCODE -ne 1) { throw "mutation not detected: $label (exit $LASTEXITCODE)" }
    Copy-Item -LiteralPath 'batch002-failure.txt' -Destination "$label.txt"
    Write-Output "DETECTED $label"
  } finally { Pop-Location }
}
if ((Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash -ne $hash) {
  throw 'production source hash changed'
}
Write-Output 'PASS 6/6 deliberately faulty isolated copies detected by differential fuzz oracle'
exit 0
