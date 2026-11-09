#!/bin/bash
# Head-node queue driver (no compute): the QoS allows only 2 submitted jobs per user, so this submits the next
# line of cluster/queue.txt whenever a slot is free, each job depending on the one before it.
#   queue.txt lines:  <afterok|afterany> <script> [args...]      (blank and # lines ignored)
#   setsid nohup bash cluster/queue.sh <last-job-id> > logs/queue.log 2>&1 < /dev/null &
cd "$HOME/vq-v2"
last=${1:?id of the job the first queued job depends on}
done_file=logs/queue.done; touch "$done_file"
while true; do
  next=$(grep -v '^\s*#' cluster/queue.txt | grep . | grep -vxF -f "$done_file" | head -1)
  [ -z "$next" ] && { echo "$(date +%T) queue empty"; exit 0; }
  if [ "$(squeue -h -u "$USER" | wc -l)" -lt 2 ]; then
    read -r dep script args <<< "$next"
    id=$(sbatch --parsable --dependency="$dep:$last" "$script" $args) || { sleep 60; continue; }
    echo "$(date +%T) submitted $id ($dep:$last) $script $args"
    echo "$next" >> "$done_file"; last=$id
  fi
  sleep 60
done
