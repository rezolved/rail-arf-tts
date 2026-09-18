# `dvc push`/`dvc pull` fail in this environment (Azure credential chain issue)

## What happened

This task's audio outputs (`data/references/{ref_single,ref_concat}.wav`,
`results/audio_samples/{comparison_set,harness,references}/`) were tracked with `dvc add` per
`CLAUDE.md`'s DVC workflow (5 `.dvc` pointer files created and committed; the raw audio itself stays
gitignored, per the repo's `**/*.wav` global ignore rule). `dvc push` (needed to upload the actual
data to `azure://ml-dvc-datasets/datasets/rail-arf-tts` so teammates/future tasks can `dvc pull` it)
does not complete in this task's local dev environment: it hangs (`timeout 60`/`timeout 30` attempts
both terminated with no output) rather than failing fast.

A related, already-documented issue surfaced earlier in this same implementation step: `dvc pull`
for t0018's `results/audio_samples/comparison_set.dvc` (needed for this task's 3-way listening-guide
comparison) failed fast with a clean `DefaultAzureCredential` error (see
`code/build_comparison_set.py`'s module docstring for the full trace) -- `EnvironmentCredential`,
`WorkloadIdentityCredential`, and `ManagedIdentityCredential` were all attempted and failed, even
though this machine has a working `az` CLI session (`az account show`/`az account get-access-token`
both succeed locally) and IMDS is reachable (confirmed via
`curl http://169.254.169.254/metadata/instance`). `dvc`'s Azure credential chain (via `adlfs`/
`azure-identity`) does not include `AzureCliCredential` in its attempted list, so it cannot reuse
the working `az` CLI session the way `az storage`/`az rest` commands on this same machine can.

`dvc push`'s hang (vs. `dvc pull`'s fast failure) suggests the push path may be retrying
`ManagedIdentityCredential` against the IMDS endpoint with a longer timeout/backoff than the pull
path did, rather than failing immediately -- not confirmed further, since reproducing it with `-v`
under a `timeout` wrapper still produced no diagnostic output before being killed.

## Impact

* This task's `.dvc` pointer files ARE committed to git (satisfying "only `.dvc` pointer files are
  committed, never raw audio blobs" -- `CLAUDE.md`'s DVC rules), so the git history is clean.
* The actual audio bytes referenced by those pointer files are **not yet uploaded** to
  `azure://ml-dvc-datasets/datasets/rail-arf-tts`. Anyone running `dvc pull` on this task's data
  right now will get the same failure this task hit pulling t0018's data, until a session with
  working Azure Blob Storage credentials (not just `az` CLI credentials) runs `dvc push` for these 5
  `.dvc` files.
* `results/listening_guide.md`'s `t0018_old_ref` column is entirely absent (documented separately in
  `code/build_comparison_set.py`'s docstring) as a direct consequence of the pull side of this same
  issue.

## Recommendation

This is an infrastructure/credential-configuration issue outside this task's own scope (`CLAUDE.md`
Key Rule 0), not a data or methodology problem. Before this task's data is durable in shared
storage, a session with working Azure Blob Storage credentials (e.g. a service-principal client
secret, or a `dvc remote modify` pointing at a credential source `dvc`'s chain actually supports)
must run:

```bash
uv run dvc push tasks/t0021_zero_shot_latency_reduction/data/references/ref_single.wav.dvc \
  tasks/t0021_zero_shot_latency_reduction/data/references/ref_concat.wav.dvc \
  tasks/t0021_zero_shot_latency_reduction/results/audio_samples/comparison_set.dvc \
  tasks/t0021_zero_shot_latency_reduction/results/audio_samples/harness.dvc \
  tasks/t0021_zero_shot_latency_reduction/results/audio_samples/references.dvc
```

Until then, the local `.dvc/cache` on THIS machine is the only copy of the pushed data; do not
assume `dvc pull` will succeed for this task's data from a different machine.

## Retry at `reporting` step (2026-09-18T23:23Z)

Per the `reporting` step's mandate to genuinely retry (not just assume prior failure still holds),
this step-executor re-ran `dvc push -v` for all 5 pointer files from this same worktree/session. The
retry did NOT hang this time (unlike the earlier `timeout 60`/`timeout 30` no-output hangs) — it
failed fast (exit 0 from the wrapping shell because `dvc` itself caught and logged the error rather
than crashing) with a clearer root cause:

```text
azure.core.exceptions.ClientAuthenticationError: DefaultAzureCredential failed to retrieve a token
from the included credentials.
Attempted credentials:
	EnvironmentCredential: environment variables are not fully configured.
	WorkloadIdentityCredential: missing required arguments 'tenant_id', 'client_id', 'token_file_path'.
	ManagedIdentityCredential: Unexpected response "{'error': 'SSO failure, to mitigated it please try
	to click Jupyter/JupyterLab.'}"
```

This confirms the earlier hypothesis: `dvc`'s Azure credential chain never reaches a credential this
Azure ML compute instance actually has configured. `az account show` works locally (interactive CLI
session for `VladimirGorovoy@rezolve.com`), but `dvc`'s chain does not include `AzureCliCredential`,
and the instance's `ManagedIdentityCredential` path fails with an Azure-ML-specific SSO error rather
than succeeding. Separately, `az storage account keys list` (an alternate path to get a usable
account key for `dvc remote modify --local azureblob account_key ...`) also failed in this
environment with `ModuleNotFoundError: No module named 'azure.mgmt.storage.operations'` — the `az`
CLI installation on this box is itself missing a required submodule for storage-account management
commands. No `AZURE_STORAGE_CONNECTION_STRING` or equivalent env var/`.env` file is present in this
worktree to fall back to.

**Conclusion**: this is still an environment/credential-configuration gap, not a data or methodology
problem, and not something resolvable by retrying the same command again. It needs either (a) a
service-principal client secret or SAS token wired into `dvc remote modify`, or (b) a working
`az storage account keys list` (which needs the `az` CLI's storage submodule fixed), or (c) an
Azure-ML-side fix to why `ManagedIdentityCredential`'s SSO path fails on this compute instance. This
remains an open, prominently-flagged item for whoever merges this task's PR — see `checkpoint.md`'s
final Next Step Notes.
