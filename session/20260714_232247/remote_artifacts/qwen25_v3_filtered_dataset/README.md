# V3 filtered training snapshot

This directory preserves the exact Brev-side SFT files used for the v3 alt-text and
heading runs. The test files are byte-identical to the frozen pre-experiment tests.

`remote_manifest_before_local_refresh.json` is the manifest copied from Brev before the
local idempotence fix. Its retained-row totals are correct, but its source-type counts
and modified-file hashes are stale. The authoritative refreshed manifest is
`tools/finetune/generated/nemo_campaign_dataset_v3/manifest.json`; it also covers the
locally reproduced reading-order, table, and aggregate-file filtering.

Use `SHA256SUMS` to verify this evidence snapshot.
