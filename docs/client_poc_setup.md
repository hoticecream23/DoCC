# Middle East client POC: setup on the client machine

The client dataset stays on the client's machine. We reach it through VPN and
remote desktop. Everything below runs on that machine. Nothing from the
dataset is copied out.

## 0. Tested on a laptop first (2026-10-22)

Cascade v2 ran end to end on a laptop (RTX 3070, 8 GB) with Ollama and
`qwen3.5:9b` (vision, Apache-2.0). Two settings are required. Without them,
every document falls back to the baseline:

| Problem | Symptom | Fix |
| --- | --- | --- |
| The model thinks before it answers and uses up `max_tokens` | `answer is not JSON` | `vlm.extra_body` in `config/cascade.yaml` turns thinking off (already set) |
| Ollama's default context is 4,096 tokens; one page needs about 5,000 | `HTTP 400` | Start Ollama with `OLLAMA_CONTEXT_LENGTH=16384`, or vLLM with `--max-model-len 16384` |

Speed on the laptop: about 20 s per document.

## 1. Find out what the machine has

Run this in PowerShell on the client machine. It only reads system facts.
Paste the output back to us.

```powershell
"OS:     " + (Get-CimInstance Win32_OperatingSystem).Caption
"CPU:    " + (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
"RAM GB: " + [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
Get-PSDrive -PSProvider FileSystem | ForEach-Object { "Disk $($_.Name): free {0:N0} GB" -f ($_.Free / 1GB) }
try { nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv } catch { "GPU: nvidia-smi not found" }
foreach ($c in "python", "py", "git", "tesseract", "ollama", "docker", "wsl") { "{0,-10} {1}" -f $c, ((Get-Command $c -ErrorAction SilentlyContinue).Source) }
foreach ($h in "pypi.org", "ollama.com", "huggingface.co", "github.com") { "{0,-15} reachable: {1}" -f $h, (Test-NetConnection $h -Port 443 -InformationLevel Quiet -WarningAction SilentlyContinue) }
```

The answers decide the path:

| Fact | Then |
| --- | --- |
| Internet reachable | Install everything directly (step 2). |
| No internet | Ask client IT to bring in the Python wheels, Tesseract, Ollama and the model weights through their approved channel. |
| GPU 8 to 16 GB | `qwen3.5:9b` on Ollama. |
| GPU 24 GB or more | Try a larger Qwen3.5 variant too, and keep the one that scores best. Check its licence first. |
| No NVIDIA GPU | Stop. Raise it with the client: the POC assumes a GPU. |

Ask the client before installing anything. Some environments forbid it.

## 2. Install

1. Python 3.10 or later.
2. Tesseract OCR, with the `eng` language. Add `ara` if the documents hold Arabic.
3. This repository (git clone, or a zip of the `small-fixes` branch), then:

   ```powershell
   pip install -e .
   ```

4. Ollama, then:

   ```powershell
   ollama pull qwen3.5:9b
   ```

## 3. Start the model server

Ollama must start with a 16k context. Quit the Ollama tray app first, then:

```powershell
$env:OLLAMA_CONTEXT_LENGTH = "16384"; ollama serve
```

Leave that window open. On a Linux GPU server, use vLLM instead, with
`--max-model-len 16384`.

## 4. Smoke test on our synthetic corpus

This proves the install before any client document is touched.

```powershell
python -m pytest -q
python -m baseline run --input corpus --output base_synth.jsonl --no-resume
python -m cascade run --input corpus --output cascade_synth.jsonl --model qwen3.5:9b --base-url http://127.0.0.1:11434/v1 --no-resume
python -m baseline compare --gold gold.jsonl --a base_synth.jsonl --b cascade_synth.jsonl
```

Expected: all tests pass. In the run log, `fallback` is 1 (the deliberately
broken `broken.pdf`). In the comparison, classification macro F1 is 1.0 for
both, and cascade metadata F1 is about 0.95.

## 5. Run on the client dataset

Write outputs outside the dataset folder. Use `D:\poc_out` or similar.

```powershell
python -m baseline profile --input <dataset> --output D:\poc_out\profile.json
python -m baseline run --input <dataset> --output D:\poc_out\base.jsonl --workers 4
python -m cascade run --input <dataset> --output D:\poc_out\cascade.jsonl --model qwen3.5:9b --base-url http://127.0.0.1:11434/v1
```

The profile gives the numbers for the Confluence page: file count, file
types, scanned against digital, and text quality.

## 6. Build gold and score

1. Export the cascade run to a review workbook:

   ```powershell
   python -m baseline export-workbook --input D:\poc_out\cascade.jsonl --output D:\poc_out\review.xlsx
   ```

2. A reviewer corrects the workbook.
3. Import the corrections as gold and compare the two pipelines:

   ```powershell
   python -m baseline import-workbook --xlsx D:\poc_out\review.xlsx --root <dataset> --output D:\poc_out\gold.jsonl
   python -m baseline compare --gold D:\poc_out\gold.jsonl --a D:\poc_out\base.jsonl --b D:\poc_out\cascade.jsonl
   ```

The class and field lists in `config/classes.yaml` and `config/fields.yaml`
come from earlier clients. Once the profile shows what this dataset holds,
update them for this client's document types.
