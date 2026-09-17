# Azure ML compute pool busy

Task `t0014_v11_decoder_fix_retrain` could not acquire a VM from `project/azure_vm.json`.

## Attempts

* **LLM-T1-NC80** (ssh_connect): SSH did not come up on LLM-T1-NC80 within deadline

## Resolution

* Check Slack (`#rail-arf-serving` for the westeurope entries, which share a NCADSH100v5 quota
  ceiling) to see whether another team is using the pool.
* Confirm each VM's state with
  `az ml compute show --name <vm-name> --resource-group <resource_group> --workspace-name <workspace> -o json`,
  reading `<resource_group>` and `<workspace>` from that VM's entry in `project/azure_vm.json` --
  the pool spans three workspaces, so there is no single correct `--workspace-name`.
* An `az_show_authorization` phase above is an RBAC problem, not a busy pool: the identity running
  `az` needs read access to the workspace's computes.
* Once a VM is free, re-run the setup-machines step.

## Resolved

`az ml compute show` confirmed LLM-T1-NC80 was mid `Stop` (last_operation Stop InProgress, started
2026-09-16T16:34:37Z) when this attempt raced it -- not a stale lock (t0010 already torn down with
cost recorded, t0013 never used the VM). Polled the VM to `state=Stopped`, then re-ran
`azure_ml_vm acquire` at 2026-09-16T16:36:51Z, which succeeded (exit 0, `ready_at`
2026-09-16T16:45:14Z). No human action was needed. See `failed_attempts[0]` in
`logs/steps/008_setup-machines/machine_log.json` for the full record.
