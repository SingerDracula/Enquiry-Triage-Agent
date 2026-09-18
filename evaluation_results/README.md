# Evaluation results

The CLI and web UI save evaluation summary JSON and per-case CSV files here by default.
Comparison artifacts are Git-ignored because new JSON results contain full reply drafts,
which may contain customer information when evaluating a non-synthetic dataset. CSV files
contain derived per-case metrics and provider failure details, but omit source email bodies
and full draft replies. Older historical JSON results do not contain saved drafts. The
historical runs were copied from the former user application-data results directory;
the originals were left unchanged.
