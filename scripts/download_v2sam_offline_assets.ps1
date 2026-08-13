param(
    [string[]] $Names = @('sam2', 'dino', 'fusion', 'vp', 'json')
)

$ErrorActionPreference = 'Stop'

$assetDir = 'D:\Code\Work\EgoExoSeg\v2sam_offline_assets'
$logDir = Join-Path $assetDir '_download_logs'
New-Item -ItemType Directory -Force -Path $assetDir, $logDir | Out-Null

$downloads = @(
    @{
        Name = 'sam2'
        File = 'sam2_hiera_large.pt'
        Url = 'https://hf-mirror.com/jaychempan/sam2/resolve/0c32b6e8de458af3f77dbaa905d66791ea375b2c/sam2_hiera_large.pt'
    },
    @{
        Name = 'dino'
        File = 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
        Url = 'https://hf-mirror.com/wangzeze/V2-SAM/resolve/50fd5a9a7e67d3fdaadab1cd0726b82896f89e02/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
    },
    @{
        Name = 'fusion'
        File = 'fusion_ego2exo_full.pth'
        Url = 'https://hf-mirror.com/wangzeze/V2-SAM/resolve/50fd5a9a7e67d3fdaadab1cd0726b82896f89e02/fusion_ego2exo_full.pth'
    },
    @{
        Name = 'vp'
        File = 'vp_ego2exo_full.pth'
        Url = 'https://hf-mirror.com/wangzeze/V2-SAM/resolve/50fd5a9a7e67d3fdaadab1cd0726b82896f89e02/vp_ego2exo_full.pth'
    },
    @{
        Name = 'json'
        File = 'json.tar.gz'
        Url = 'https://hf-mirror.com/wangzeze/V2-SAM/resolve/50fd5a9a7e67d3fdaadab1cd0726b82896f89e02/json.tar.gz'
    }
)

foreach ($item in $downloads) {
    if ($item.Name -notin $Names) {
        continue
    }
    $outputPath = Join-Path $assetDir $item.File
    $stdoutPath = Join-Path $logDir ($item.Name + '.stdout.log')
    $stderrPath = Join-Path $logDir ($item.Name + '.stderr.log')
    $pidPath = Join-Path $logDir ($item.Name + '.pid')

    $arguments = @(
        '--fail', '--location', '--silent', '--show-error',
        '--retry', '100', '--retry-all-errors', '--retry-delay', '2',
        '--connect-timeout', '20', '--speed-limit', '1024', '--speed-time', '30',
        '--continue-at', '-',
        '--output', $outputPath, $item.Url
    )
    $process = Start-Process -FilePath 'curl.exe' -ArgumentList $arguments `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath `
        -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath $pidPath -Value $process.Id -NoNewline
    [pscustomobject]@{
        Name = $item.Name
        PID = $process.Id
        Output = $outputPath
    }
}
