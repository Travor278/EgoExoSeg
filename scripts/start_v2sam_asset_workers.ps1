$ErrorActionPreference = 'Stop'

$assetDir = 'D:\Code\Work\EgoExoSeg\v2sam_offline_assets'
$logDir = Join-Path $assetDir '_download_logs'
$workerScript = 'D:\Code\Work\EgoExoSeg\scripts\download_v2sam_asset_worker.ps1'
New-Item -ItemType Directory -Force -Path $assetDir, $logDir | Out-Null

$assets = @(
    @{ Name='sam2'; File='sam2_hiera_large.pt'; Bytes=897952466L; Sha='7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b'; Url='https://hf-mirror.com/jaychempan/sam2/resolve/0c32b6e8de458af3f77dbaa905d66791ea375b2c/sam2_hiera_large.pt' },
    @{ Name='dino'; File='dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'; Bytes=1213050671L; Sha='8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035'; Url='https://hf-mirror.com/wangzeze/V2-SAM/resolve/50fd5a9a7e67d3fdaadab1cd0726b82896f89e02/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth' },
    @{ Name='fusion'; File='fusion_ego2exo_full.pth'; Bytes=2132357587L; Sha='f0c986c0296c3eee9a64ee5fef9f48c60e0b1de28a94cf711f6037824da6eddb'; Url='https://hf-mirror.com/wangzeze/V2-SAM/resolve/50fd5a9a7e67d3fdaadab1cd0726b82896f89e02/fusion_ego2exo_full.pth' },
    @{ Name='vp'; File='vp_ego2exo_full.pth'; Bytes=922106282L; Sha='80b3a2ab59b9453734aac2060f9c8a2f530b548b6f312f87efd7e9545aa9e3eb'; Url='https://hf-mirror.com/wangzeze/V2-SAM/resolve/50fd5a9a7e67d3fdaadab1cd0726b82896f89e02/vp_ego2exo_full.pth' },
    @{ Name='json'; File='json.tar.gz'; Bytes=1333324828L; Sha='1852df7d8aa6bc53f15fb05d129105131a67d53f49ca92705639ee32119dc74f'; Url='https://hf-mirror.com/wangzeze/V2-SAM/resolve/50fd5a9a7e67d3fdaadab1cd0726b82896f89e02/json.tar.gz' }
)

foreach ($asset in $assets) {
    $outputPath = Join-Path $assetDir $asset.File
    $stdoutPath = Join-Path $logDir ($asset.Name + '.worker.stdout.log')
    $stderrPath = Join-Path $logDir ($asset.Name + '.worker.stderr.log')
    $pidPath = Join-Path $logDir ($asset.Name + '.worker.pid')
    $arguments = @(
        '-NoProfile', '-File', $workerScript,
        '-Name', $asset.Name,
        '-OutputPath', $outputPath,
        '-Url', $asset.Url,
        '-ExpectedBytes', $asset.Bytes,
        '-ExpectedSha256', $asset.Sha,
        '-LogDir', $logDir
    )
    $process = Start-Process -FilePath 'pwsh.exe' -ArgumentList $arguments `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath `
        -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath $pidPath -Value $process.Id -NoNewline
    [pscustomobject]@{ Name=$asset.Name; PID=$process.Id; StartBytes=(Get-Item -LiteralPath $outputPath).Length }
}
