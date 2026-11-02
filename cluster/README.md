# Running VeriQuery's LLM on the CCDS GPU cluster (TC2)

The cluster serves the model; the pipeline keeps running on your laptop and talks to it through
an SSH tunnel. Follows the CCDS TC2 user guide: jobs go through SLURM (`sbatch`), never run on
the head node, and the `normal` QoS allows 1 GPU, 10 CPUs, 30 GB RAM and 6 h per job.

## 0. Before you start (once)

- Connect to the NTU VPN (https://ntuvpn.ntu.edu.sg) when off campus. TC2 is at `10.96.189.12`.
- Set up key login so scripts (and Claude Code) can run commands without your password.
  In PowerShell on your laptop:

  ```powershell
  ssh-keygen -t ed25519                     # accept defaults
  type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh <ntu_id>@10.96.189.12 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
  ssh <ntu_id>@10.96.189.12 mytcinfo        # should print your QoS without asking for a password
  ```

## 1. Copy the repo and set up the environment (head node)

```bash
# from your laptop, in the repo folder
scp -r veriquery cluster benchmarks examples pyproject.toml README.md <ntu_id>@10.96.189.12:~/veriquery/
ssh <ntu_id>@10.96.189.12
cd ~/veriquery && bash cluster/setup_env.sh        # conda env + vLLM + model download (~20 GB)
```

## 2. Check the GPU, then start the model server

```bash
sbatch cluster/check_gpu.sbatch && sleep 60 && cat cluster/logs/vq-gpu_*.out
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct-AWQ sbatch cluster/serve_llm.sbatch
squeue -u $USER                                     # wait for state R
grep VERIQUERY_ENDPOINT cluster/logs/vq-llm_*.out   # e.g. node=10.128.10.31 port=8000
```

Model size by GPU memory: 48 GB (A40/A6000) → 32B AWQ; 24 GB → 14B AWQ or 32B AWQ with
`--max-model-len 8192`; 16 GB → 7B/8B.

## 3. Tunnel from your laptop and run VeriQuery

```powershell
ssh -N -L 8000:<node-ip>:8000 <ntu_id>@10.96.189.12      # leave this window open

$env:VERIQUERY_LLM_PROVIDER="openai"
$env:VERIQUERY_LLM_BASE_URL="http://localhost:8000/v1"
python examples/retail_demo.py --model Qwen/Qwen2.5-Coder-32B-Instruct-AWQ
```

The job ends after 6 h; resubmit step 2 (the tunnel's node IP may change). `scancel <jobid>`
stops it early; please do, the GPU is shared.
