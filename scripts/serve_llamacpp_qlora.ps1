# Serve the base Qwen2.5-Coder-7B (4-bit) plus the QLoRA writer adapter on a laptop GPU with llama.cpp.
# Used by the `laptop_qlora` profile in configs/models.yaml (endpoint `llamacpp`, port 8081).
#
# One-time setup (see experiments/local_ollama_base7b/README.md):
#   1. Download a llama.cpp Windows CUDA release (llama-bXXXX-bin-win-cuda-*-x64.zip + matching cudart zip).
#   2. Convert the adapter:  python convert_lora_to_gguf.py --outtype f16 --outfile writer-qlora-f16.gguf <adapter dir>
#   3. `ollama pull qwen2.5-coder:7b` (its GGUF blob is reused as the base model).
# Then: powershell -File scripts/serve_llamacpp_qlora.ps1
# and in another terminal: $env:VERIQUERY_PROFILE="laptop_qlora"; python -m uvicorn veriquery.server.app:app --port 8080
#
# The adapter is loaded once as id 0; each request sets its scale (1.0 for the writer, 0.0 for the other agents).
param(
    [string]$LlamaDir = "$env:USERPROFILE\tools\llama.cpp-b11972",
    [string]$Adapter = "$env:USERPROFILE\tools\llama.cpp-b11972\writer-qlora-f16.gguf",
    [string]$BaseGguf = "",
    [int]$Port = 8081,
    [int]$Ctx = 24576,      # shared by the parallel slots: 3 x 8192 tokens
    [int]$Parallel = 3
)
if (-not $BaseGguf) {
    $manifest = "$env:USERPROFILE\.ollama\models\manifests\registry.ollama.ai\library\qwen2.5-coder\7b"
    $digest = ((Get-Content $manifest -Raw | ConvertFrom-Json).layers | Where-Object { $_.mediaType -eq "application/vnd.ollama.image.model" }).digest
    $BaseGguf = "$env:USERPROFILE\.ollama\models\blobs\" + ($digest -replace ":", "-")
}
# Free the GPU if Ollama still holds the same model.
try { & ollama stop qwen2.5-coder:7b 2>$null } catch {}
& "$LlamaDir\bin\llama-server.exe" -m $BaseGguf --lora $Adapter --lora-init-without-apply `
    --host 127.0.0.1 --port $Port -ngl 99 -c $Ctx -np $Parallel -fa on -ctk q8_0 -ctv q8_0 --jinja
