# OpenBox NDA compiler

Upload an NDA, get enforceable OpenBox policies on the agents that handle the
counterparty's documents — each rule traceable to the clause that required it,
in a few seconds.

```
parse → classify → extract → build → verify → apply
 code    JEV ×1     OpenAI ×n  code    JEV ×n   OpenBox API ×n
```

| Stage | What | Who |
|---|---|---|
| parse | PDF/DOCX/MD → numbered clauses, definitions section isolated | code |
| classify | one call, whole NDA as state, a `choice` per clause → control kind | TypeSafe JEV |
| extract | fill a fixed obligation form; every literal must appear verbatim in the NDA | OpenAI (`gpt-5-mini`, strict JSON schema) |
| build | obligation × bindings table → Policy Rule / Behavior Rule / Guardrail / judgement payloads | code |
| verify | `noul`: does this rule faithfully enforce this clause? below threshold → review | TypeSafe JEV |
| apply | create inactive → `/evaluate` block + allow case → activate | OpenBox backend |

Clauses that cannot be enforced on an agent's tool calls (term, return or
destroy, governing law) are counted in the coverage figure, never dropped
silently.

## Run

```bash
uv sync
cp .env.example .env            # leave keys blank for an offline dry-run
uv run nda-compile fixtures/coca_cola_nda.md --bindings bindings/coca-cola.yaml
uv run nda-web                  # http://127.0.0.1:8010
```

Any stage whose key is missing runs a deterministic fake, so the pipeline
works end to end with no credentials and the payloads it would send can be
read before an org key exists.

## Bindings

The NDA says "Representatives" and "the Disclosing Party"; OpenBox rules
match on agent ids and folder prefixes. `bindings/<matter>.yaml` is where the
compliance team records that mapping once per matter. See
`bindings/coca-cola.yaml`, which targets the agents in `openbox-barrier-demo`.

## Governing the compiler

Set `COMPILER_OPENBOX_API_KEY` and the graph itself runs under OpenBox. A
REQUIRE_APPROVAL rule on the `apply` activity pauses the run until someone
approves the new rules.

## Tests

```bash
uv run pytest
```
