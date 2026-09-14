# Metrics (3)

## 📊 ratio (2)

<details>
<summary>📊 <strong>Real-Time Factor</strong> (<code>rtf</code>)</summary>

| Field | Value |
|---|---|
| **Key** | `rtf` |
| **Unit** | ratio |
| **Value type** | float |
| **Datasets** | — |

Synthesis wall time divided by output audio duration. RTF < 1.0 means faster than real time.
Lower is better.

</details>

<details>
<summary>📊 <strong>Speaker Similarity (GE2E Cosine)</strong>
(<code>speaker_sim</code>)</summary>

| Field | Value |
|---|---|
| **Key** | `speaker_sim` |
| **Unit** | ratio |
| **Value type** | float |
| **Datasets** | — |

Cosine similarity between synthesized clip GE2E embedding and mean GE2E embedding of
ElevenLabs David reference set. Computed via resemblyzer. Higher is better.

</details>

## 📏 ms (1)

<details>
<summary>📏 <strong>Time to First Byte (ms)</strong> (<code>ttfb_ms</code>)</summary>

| Field | Value |
|---|---|
| **Key** | `ttfb_ms` |
| **Unit** | ms |
| **Value type** | float |
| **Datasets** | — |

Wall-clock milliseconds from TTS request issue to first audio byte received. Measured locally
(same-machine or loopback). Report p50, p95, p99. Lower is better.

</details>
