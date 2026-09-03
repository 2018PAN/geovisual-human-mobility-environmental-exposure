param(
    [switch]$Overwrite,
    [switch]$SkipSpatialCV
)

$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "python"
$DecompositionInput = Join-Path (
    Split-Path -Parent $CodeRoot
) "Output\Decomposition\grid_level_decomposition.parquet"
$ComponentMoran = Join-Path (
    Split-Path -Parent $CodeRoot
) "Output\SpatialAnalysis\GlobalMoran\component_global_moran_results.csv"
$ComponentVariables = @(
    "festival_pre_mobility_component",
    "festival_pre_pollution_component",
    "post_festival_mobility_component",
    "post_festival_pollution_component"
)

function Invoke-Stage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Script,
        [string[]]$Arguments = @()
    )
    $ScriptPath = Join-Path $CodeRoot $Script
    if (-not (Test-Path -LiteralPath $ScriptPath)) {
        throw "Required script not found: $ScriptPath"
    }
    $StageArguments = @($ScriptPath) + $Arguments
    if ($Overwrite) {
        $StageArguments += "--overwrite"
    }
    Write-Host "RUN: $Python $($StageArguments -join ' ')"
    & $Python @StageArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stage failed with exit code ${LASTEXITCODE}: $ScriptPath"
    }
}

$GlobalMoranArguments = @(
    "--input-path", $DecompositionInput,
    "--output-name", "component_global_moran_results.csv",
    "--variables"
) + $ComponentVariables
Invoke-Stage "06_spatial_analysis\01_global_moran.py" $GlobalMoranArguments

$LisaArguments = @(
    "--input-path", $DecompositionInput,
    "--variables"
) + $ComponentVariables
Invoke-Stage "06_spatial_analysis\02_local_moran_lisa.py" $LisaArguments

$SpatialPlotArguments = @(
    "--global-results", $ComponentMoran,
    "--global-figure-name", "component_global_moran_results.png",
    "--variables"
) + $ComponentVariables
Invoke-Stage "06_spatial_analysis\03_plot_spatial_analysis.py" $SpatialPlotArguments

foreach ($Target in @("mobility", "pollution")) {
    Invoke-Stage "08_modeling\01_build_xgboost_input.py" @(
        "--comparison", "all",
        "--analysis-target", $Target
    )
    $ModelArguments = @(
        "--comparison", "all",
        "--analysis-target", $Target
    )
    if ($SkipSpatialCV) {
        $ModelArguments += "--skip-spatial-cv"
    }
    Invoke-Stage "08_modeling\02_run_xgboost.py" $ModelArguments
    Invoke-Stage "08_modeling\03_calculate_shap.py" @(
        "--comparison", "all",
        "--analysis-target", $Target
    )
    Invoke-Stage "08_modeling\04_plot_xgboost_shap.py" @(
        "--comparison", "all",
        "--analysis-target", $Target
    )
}

Write-Host "Mobility/pollution component analyses completed."
