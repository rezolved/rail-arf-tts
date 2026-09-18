# Azure ML compute pool busy

Task `t0021_zero_shot_latency_reduction` could not acquire a VM from `project/azure_vm.json`.

## Attempts

* **LLM-T1-NC80** (ssh_connect): SSH did not come up on LLM-T1-NC80 within deadline

## Resolution

* Check Slack (`#rail-arf-serving` for the westeurope entries, which share a
  NCADSH100v5 quota ceiling) to see whether another team is using the pool.
* Confirm each VM's state with `az ml compute show --name <vm-name> --resource-group <resource_group> --workspace-name <workspace> -o json`,
  reading `<resource_group>` and `<workspace>` from that VM's entry in
  `project/azure_vm.json` -- the pool spans three workspaces, so there is no
  single correct `--workspace-name`.
* An `az_show_authorization` phase above is an RBAC problem, not a busy pool:
  the identity running `az` needs read access to the workspace's computes.
* Once a VM is free, re-run the setup-machines step.
