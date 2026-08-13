$ErrorActionPreference = 'Stop'

$sourceDir = 'D:\Code\Work\EgoExoSeg\v2sam_offline_assets'
$outputDir = Join-Path $sourceDir 'upload_parts_700MiB'
$chunkBytes = 700MB
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$files = @(
    'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
    'fusion_ego2exo_full.pth',
    'json.tar.gz'
)

foreach ($fileName in $files) {
    $sourcePath = Join-Path $sourceDir $fileName
    $input = [System.IO.File]::OpenRead($sourcePath)
    try {
        $partNumber = 0
        $buffer = New-Object byte[] (8MB)
        while ($input.Position -lt $input.Length) {
            $partName = '{0}.part{1:d3}' -f $fileName, $partNumber
            $partPath = Join-Path $outputDir $partName
            $remaining = [Math]::Min([long]$chunkBytes, $input.Length - $input.Position)
            $output = [System.IO.File]::Create($partPath)
            try {
                while ($remaining -gt 0) {
                    $count = [Math]::Min([long]$buffer.Length, $remaining)
                    $read = $input.Read($buffer, 0, [int]$count)
                    if ($read -le 0) { throw "Unexpected EOF in $fileName" }
                    $output.Write($buffer, 0, $read)
                    $remaining -= $read
                }
                $output.Flush($true)
            } finally {
                $output.Dispose()
            }
            $partNumber++
        }
    } finally {
        $input.Dispose()
    }
}

$manifestPath = Join-Path $outputDir 'SHA256SUMS.parts.txt'
$lines = foreach ($part in Get-ChildItem -LiteralPath $outputDir -File |
        Where-Object Name -like '*.part*' | Sort-Object Name) {
    $sha = (Get-FileHash -Algorithm SHA256 -LiteralPath $part.FullName).Hash.ToLowerInvariant()
    '{0}  {1}' -f $sha, $part.Name
}
[System.IO.File]::WriteAllLines($manifestPath, $lines, [System.Text.UTF8Encoding]::new($false))

Get-ChildItem -LiteralPath $outputDir -File | Sort-Object Name |
    Select-Object Name, Length
