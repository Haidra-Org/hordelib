# Ad-hoc download controls

LoRA and textual-inversion managers expose `configure_downloads(connections=4)`. Call it after
construction to set the maximum connections per eligible large file, clamped to 1 to 8. An active
segmented transfer keeps the starting connection count. Queue order, file admission and pool size
do not change. One large LoRA can use the configured connections, and several files can still be
fetched in parallel through the pool.

Direct consumers that do not configure the managers keep the defaults, including
`AIWORKER_LORA_DOWNLOAD_CONNECTIONS`. The worker can override these with the shared per-file
connection setting. The setting is a per-file count, and the files do not share a socket budget.
Four concurrent files with four segments each can open sixteen range requests. Remote-host limits
and local storage contention can affect the resulting throughput.

`adhoc.download_success` includes the end-to-end duration and `commit_seconds`, the time spent on
cache eviction, reference merge and save. For segmented files, the horde-model-reference engine
reports transfer and checksum timing separately. Engine transfer summaries include the number of
connections actually used. Compare the same file, credentials, destination volume and inference
workload before attributing a slow success duration to network bandwidth.
