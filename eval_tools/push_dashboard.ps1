$pair = "admin:admin"
$bytes = [System.Text.Encoding]::ASCII.GetBytes($pair)
$base64 = [Convert]::ToBase64String($bytes)
$headers = @{
    Authorization = "Basic $base64"
    "Content-Type" = "application/json"
}

$body = Get-Content -Raw -Encoding UTF8 "eval_tools/dashboard.json"
$response = Invoke-RestMethod -Uri "http://localhost:3000/api/dashboards/db" -Method Post -Headers $headers -Body $body
$response | ConvertTo-Json
