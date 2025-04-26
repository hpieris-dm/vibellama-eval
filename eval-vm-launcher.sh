#!/usr/bin/env bash
set -euo pipefail

# ─── CONFIG ────────────────────────────────────────────────────────────────
PROJECT="rnd-sentiment-llm"
ZONE="us-central1-f"
MACHINE_TYPE="a2-highgpu-1g"
NETWORK_IF="network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default"
ACCELERATOR="count=1,type=nvidia-tesla-a100"
IMAGE="projects/rnd-sentiment-llm/global/images/llm-tocrch-cuda-ft"
DISK_OPTS="auto-delete=yes,boot=yes,device-name=%s,image=${IMAGE},mode=rw,size=150,type=pd-balanced"
# shutdown after 3 days = 4320 minutes
SHUTDOWN_CMD="sudo shutdown -h +4320"

# ─── LOOP & CREATE ─────────────────────────────────────────────────────────
for i in 1 2 3; do
  NAME=$(printf "vibellama-eval-%02d" "$i")
  echo "[+] Creating VM: $NAME"
  gcloud compute instances create "$NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --network-interface="$NETWORK_IF" \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --accelerator="$ACCELERATOR" \
    --create-disk=$(printf "$DISK_OPTS" "$NAME") \
    --metadata=startup-script="$SHUTDOWN_CMD" \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring

  echo "[✓]  Launched $NAME (will auto-shutdown in 3 days)"
done

echo "All 3 instances launched."
