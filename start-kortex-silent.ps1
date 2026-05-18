#!/usr/bin/env pwsh
# Kortex Silent Background Starter
# Automatically verifies Ollama & starts Kortex AIM Proxy completely hidden in the background.

$KortexRoot = "C:\Users\HADES\Desktop\CodeSigil\kortex"
$ProxyPath = "$KortexRoot\target\release\aim-proxy.exe"

# 1. Ensure Ollama is running silently
try {
    $null = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 -ErrorAction Stop
} catch {
    # If not running, start Ollama serve in hidden window style
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

# 2. Ensure Kortex AIM Proxy is running silently
try {
    $null = Invoke-WebRequest -Uri "http://127.0.0.1:1536/" -TimeoutSec 3 -ErrorAction Stop
} catch {
    # If not running, start AIM Proxy in hidden window style with correct working directory
    if (Test-Path $ProxyPath) {
        Start-Process $ProxyPath -WorkingDirectory $KortexRoot -WindowStyle Hidden
    }
}
