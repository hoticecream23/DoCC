# Middle East client POC: setup on the client machines

The client dataset stays inside the client's network. Nothing from it is
copied out.

```
laptop --OpenVPN--> client network --Remote Desktop--> VM (no GPU)
                                                        |  pipeline, data, Tesseract
                                                        |  HTTP on the local network
                                                        v
                                                     GPU server
                                                        model server (Ollama or vLLM)
```

- The **VM** holds the dataset and runs the pipeline (`baseline`, `cascade`).
  It has no NVIDIA GPU.
- The **GPU server** runs only the model server. The client's sheet gives its
  local and public IP. Use the **local IP** (10.x, 172.16-31.x or 192.168.x).
  It works only while OpenVPN is connected.
- Page images and page text go from the VM to the GPU server. Both machines
  are inside the client network, so the data stays on the client side.

## 0. Tested on a laptop first (2026-10-22)

Cascade v2 ran end to end on a laptop (RTX 3070, 8 GB) with Ollama and
`qwen3.5:9b` (vision, Apache-2.0). Two settings are required. Without them,
every document falls back to the baseline:

| Problem | Symptom | Fix |
| --- | --- | --- |
| The model thinks before it answers and uses up `max_tokens` | `answer is not JSON` | `vlm.extra_body` in `config/cascade.yaml` turns thinking off (already set) |
| Ollama's default context is 4,096 tokens; one page needs about 5,000 | `HTTP 400` | Start Ollama with `OLLAMA_CONTEXT_LENGTH=16384`, or vLLM with `--max-model-len 16384` |

Speed on the laptop: about 20 s per document.

## 1. Find out what the machines have

### 1a. The VM

Run this in PowerShell on the VM. It only reads system facts. Paste the
output back to us.

```powershell
"OS:     " + (Get-CimInstance Win32_OperatingSystem).Caption
"CPU:    " + (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
"RAM GB: " + [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
Get-PSDrive -PSProvider FileSystem | ForEach-Object { "Disk $($_.Name): free {0:N0} GB" -f ($_.Free / 1GB) }
foreach ($c in "python", "py", "git", "tesseract", "ssh") { "{0,-10} {1}" -f $c, ((Get-Command $c -ErrorAction SilentlyContinue).Source) }
foreach ($h in "pypi.org", "github.com", "ollama.com", "huggingface.co") { "{0,-15} reachable: {1}" -f $h, (Test-NetConnection $h -Port 443 -InformationLevel Quiet -WarningAction SilentlyContinue) }
```

### 1b. The GPU server

From PowerShell on the VM, connect with the username from the sheet:

```powershell
ssh <username>@<gpu-local-ip>
```

If it is a Linux server, run this there and paste the output back:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
grep PRETTY_NAME /etc/os-release; free -g | head -2; df -h ~ | tail -1
which python3 docker ollama vllm
curl -sI https://ollama.com | head -1; curl -sI https://huggingface.co | head -1
```

If `ssh` is refused, ask the client how they expect you to reach the GPU
server (SSH key, Remote Desktop, or a web console).

The answers decide the path:

| Fact | Then |
| --- | --- |
| Internet reachable | Install directly (steps 2 and 3). |
| No internet | Ask client IT to bring in the Python wheels, Tesseract, Ollama and the model weights through their approved channel. |
| GPU 8 to 16 GB | `qwen3.5:9b`. `ministral-3:14b` fits from about 12 GB. |
| GPU 24 GB or more | Also try Qwen3.8 27B. Keep the model that scores best. |

Check each model's licence on its model card before the client run.
Ask the client before installing anything. Some environments forbid it.

## 2. Install the pipeline on the VM

1. Python 3.10 or later.
2. Tesseract OCR, with the `eng` language. Add `ara` if the documents hold Arabic.
3. The code: extract `DoCC_small-fixes.zip`, or `git clone -b small-fixes` the
   repository. Then, in the code folder:

   ```powershell
   pip install -e . pytest
   ```

## 3. Start the model server on the GPU server

The model server must listen on the network, not only on the GPU server
itself, so that the VM can reach it.

### Option A: Ollama (tested)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:9b
OLLAMA_HOST=0.0.0.0:11434 OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```

Ollama has no password. Anyone on the client network who can reach port
11434 can use the model. Ask client IT to allow that port from the VM only.

If the install script set Ollama up as a system service, stop it first
(`sudo systemctl stop ollama`), or put the two settings in the service with
`sudo systemctl edit ollama`.

### Option B: vLLM (supports an API key)

```bash
export DOCC_VLM_KEY=<a long random string>
vllm serve <hugging-face-model-id> --host 0.0.0.0 --port 8000 --max-model-len 16384 --api-key "$DOCC_VLM_KEY"
```

## 4. Point the VM at the GPU server

Edit `config/cascade.yaml` on the VM:

```yaml
vlm:
  base_url: http://<gpu-local-ip>:11434/v1    # vLLM: port 8000
  model: qwen3.5:9b                           # the name the server uses
  api_key_env: null                           # vLLM: DOCC_VLM_KEY
  allowed_hosts: ["<gpu-local-ip>"]
```

`allowed_hosts` is required. Cascade refuses any model server that is not
on the same machine, unless it is listed here. This guard stops data being
sent to an unintended host.

For vLLM, set the same key on the VM before each run:

```powershell
$env:DOCC_VLM_KEY = "<the same string>"
```

Check that the VM reaches the server:

```powershell
Test-NetConnection <gpu-local-ip> -Port 11434
curl.exe http://<gpu-local-ip>:11434/v1/models
```

The second command must list the model. If it times out, a firewall blocks
the port. Ask client IT to open it from the VM to the GPU server.

## 5. Smoke test on our synthetic corpus

This proves the install before any client document is touched. Run it on
the VM, in the code folder.

```powershell
python -m pytest -q
python tests/make_corpus.py corpus
python tests/make_gold.py gold.jsonl
python -m baseline run --input corpus --output base_synth.jsonl --no-resume
python -m cascade run --input corpus --output cascade_synth.jsonl --no-resume
python -m baseline compare --gold gold.jsonl --a base_synth.jsonl --b cascade_synth.jsonl
```

The second and third lines build the 10-document synthetic corpus and its
gold. They are not in the zip or in git.

Expected: all tests pass. In the run log, `fallback` is 1 (the deliberately
broken `broken.pdf`). In the comparison, classification macro F1 is 1.0 for
both, and cascade metadata F1 is about 0.95.

## 6. Run on the client dataset

Write outputs outside the dataset folder. Use `D:\poc_out` or similar.

```powershell
python -m baseline profile --input <dataset> --output D:\poc_out\profile.json
python -m baseline run --input <dataset> --output D:\poc_out\base.jsonl --workers 4
python -m cascade run --input <dataset> --output D:\poc_out\cascade.jsonl
```

The profile gives the numbers for the Confluence page: file count, file
types, scanned against digital, and text quality.

To try another model, pass `--model <name>` and write to another output file.

## 7. Build gold and score

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
