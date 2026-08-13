param(
    [Parameter(Mandatory = $true)] [string] $Name,
    [Parameter(Mandatory = $true)] [string] $OutputPath,
    [Parameter(Mandatory = $true)] [string] $Url,
    [Parameter(Mandatory = $true)] [long] $ExpectedBytes,
    [Parameter(Mandatory = $true)] [string] $ExpectedSha256,
    [Parameter(Mandatory = $true)] [string] $LogDir
)

$ErrorActionPreference = 'Stop'
$statusPath = Join-Path $LogDir ($Name + '.status')
$workerLog = Join-Path $LogDir ($Name + '.worker.log')
$curlErr = Join-Path $LogDir ($Name + '.curl.stderr.log')

Set-Content -LiteralPath $statusPath -Value 'RUNNING'

while ($true) {
    $currentBytes = if (Test-Path -LiteralPath $OutputPath) {
        (Get-Item -LiteralPath $OutputPath).Length
    } else {
        0L
    }

    if ($currentBytes -gt $ExpectedBytes) {
        Set-Content -LiteralPath $statusPath -Value "FAILED_OVERSIZE:$currentBytes"
        exit 2
    }

    if ($currentBytes -eq $ExpectedBytes) {
        $actualSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $OutputPath).Hash.ToLowerInvariant()
        if ($actualSha -ne $ExpectedSha256.ToLowerInvariant()) {
            Set-Content -LiteralPath $statusPath -Value "FAILED_SHA256:$actualSha"
            exit 3
        }
        Set-Content -LiteralPath $statusPath -Value "VERIFIED:$actualSha"
        exit 0
    }

    $before = $currentBytes
    & curl.exe --fail --location --silent --show-error `
        --connect-timeout 20 --speed-limit 1024 --speed-time 30 --max-time 90 `
        --continue-at - --output $OutputPath $Url 2>> $curlErr
    $curlExit = $LASTEXITCODE

    $after = if (Test-Path -LiteralPath $OutputPath) {
        (Get-Item -LiteralPath $OutputPath).Length
    } else {
        0L
    }
    Add-Content -LiteralPath $workerLog -Value `
        ("{0:o}`texit={1}`tbefore={2}`tafter={3}`tdelta={4}" -f `
            (Get-Date), $curlExit, $before, $after, ($after - $before))

    if ($after -lt $before) {
        Set-Content -LiteralPath $statusPath -Value "FAILED_REGRESSION:$before->$after"
        exit 4
    }
    Start-Sleep -Seconds 2
}
