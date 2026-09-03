param(
    [switch]$Overwrite,
    [switch]$SkipSpatialCV
)

$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "python"

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

Invoke-Stage "05_aggregation\01_build_hourly_analysis_dataset.py" @(
    "--start-date", "2018-02-01",
    "--end-date", "2018-03-12"
)
Invoke-Stage "05_aggregation\02_aggregate_chunyun_periods.py"
Invoke-Stage "05_aggregation\03_calculate_period_changes.py"
Invoke-Stage "06_spatial_analysis\01_global_moran.py"
Invoke-Stage "06_spatial_analysis\02_local_moran_lisa.py"
Invoke-Stage "06_spatial_analysis\03_plot_spatial_analysis.py"
Invoke-Stage "07_decomposition\01_prepare_decomposition_input.py"
Invoke-Stage "07_decomposition\02_run_exposure_decomposition.py"
Invoke-Stage "07_decomposition\03_plot_decomposition.py"
Invoke-Stage "08_modeling\01_build_xgboost_input.py" @("--comparison", "all")

$ModelArguments = @("--comparison", "all")
if ($SkipSpatialCV) {
    $ModelArguments += "--skip-spatial-cv"
}
Invoke-Stage "08_modeling\02_run_xgboost.py" $ModelArguments
Invoke-Stage "08_modeling\03_calculate_shap.py" @("--comparison", "all")
Invoke-Stage "08_modeling\04_plot_xgboost_shap.py" @("--comparison", "all")
Invoke-Stage "09_visualization\01_plot_hourly_timeseries.py"
Invoke-Stage "09_visualization\02_plot_period_maps.py"
Invoke-Stage "09_visualization\03_plot_change_maps.py"
Invoke-Stage "09_visualization\04_plot_summary_figures.py"

Write-Host "Downstream Chunyun pipeline completed."
