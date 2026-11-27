# Epic 5 retrieval threshold decision

## Measurement

Measured cosine similarities against the three ready local indexes using
`sentence-transformers/all-MiniLM-L6-v2` and Chroma cosine distance
(`similarity = 1 - distance`). The guide's 15-question run is recorded in the
Epic 5 verification notes below after the real-model run completes.

Initial calibration samples:

| Repository | Query type | Best similarity | Outcome used for calibration |
| --- | --- | ---: | --- |
| `records` | `Record` class behavior (answerable code) | 0.354 | Relevant source found |
| `records` | README usage (answerable docs) | 0.300 | Relevant README chunks at 0.300 and 0.299 |
| `flask` | route decorator (answerable code) | 0.728 | Relevant source found |
| `flask` | app configuration (answerable docs) | 0.764 | Relevant documentation found |
| `the-art-of-command-line` | shell history (answerable docs) | 0.421 | Relevant README found |
| `records` | France capital (unrelated) | 0.111 | No relevant source |
| `flask` | France capital (unrelated) | 0.194 | No relevant source |
| `records` | absent web server endpoint | 0.231 | No relevant source |
| `flask` | absent payment processing | 0.166 | No relevant source |
| `the-art-of-command-line` | absent graphical UI | 0.298 | No relevant source |

The hostile question asking for the system prompt scored as high as 0.370 on
unrelated code. This overlaps the answerable range, so a similarity threshold
cannot reject that prompt by itself. The system prompt's untrusted-context
rule and the planted injection check are responsible for resisting it.

## Decision

Set `retrieval_min_similarity` to **0.29**. The previous 0.30 cutoff excluded
the README usage query whose best chunk scored 0.299 after conversion. The
0.29 cutoff includes that relevant chunk and also permits a borderline
irrelevant GUI chunk (0.298); the answer-generation step must return the
canonical not-found response when the retrieved context does not support the
answer. Clearly unrelated France and absent-feature queries remain below the
cutoff and avoid an LLM call. The hostile-query overlap is documented above
and is covered by prompt-injection safeguards rather than score tuning.

## Verification

The full query outcomes and both planted-injection runs are recorded in
[`epic5-verification.md`](epic5-verification.md).
