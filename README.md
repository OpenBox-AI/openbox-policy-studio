# OpenBox NDA compiler

Upload an NDA, get enforceable OpenBox policies on the agents that handle the
counterparty's documents — each rule traceable to the clause that required it,
in a few seconds.

```
parse → classify → extract → build → verify → apply
 code    Decisions   Claude ×n  code    Decisions   OpenBox API ×n
```

| Stage | What | Who |
|---|---|---|
| parse | PDF/DOCX/MD → numbered clauses, definitions section isolated; a repeated number (an amendment restarting at 1) gets a unique id, `2~2` | code |
| classify | a `choice` per clause → control kind, the whole NDA as shared input; then the **recipient check**, a yes/no per remaining clause: does it limit who may receive the information? | OpenAI Decisions API (`gpt-6-luna`), or TypeSafe JEV |
| extract | fill a fixed obligation form, plus three judgements: does consent lift the duty, is its breach itself material, does the clause really impose this duty. Every value a rule is built from must appear verbatim in the NDA | Claude (`claude-sonnet-5-5`, structured JSON), or OpenAI |
| build | obligation × bindings → policy rules; one rule per condition set, strictest decision wins | code |
| verify | yes/no: would enforcing this rule be a mistake under its clause? below threshold → review | OpenAI Decisions API, or TypeSafe JEV |
| apply | create inactive → `/evaluate` block + allow case → activate, three rules at a time | OpenBox backend |

The decision comes from the extraction: REQUIRE_APPROVAL when consent lifts
the duty (in any wording), HALT when the clause makes its own breach material
(or a sub-clause says so about it), otherwise BLOCK.

Clauses that cannot be enforced on an agent's tool calls (term, return or
destroy, governing law) are counted in the coverage figure, never dropped
silently.

## Keys

| Key | Used for | Without it |
|---|---|---|
| `ANTHROPIC_API_KEY` | extraction with Claude Sonnet 5.5 (`ANTHROPIC_MODEL`) | extraction falls back to OpenAI (`OPENAI_MODEL`, gpt-5-mini) |
| `OPENAI_API_KEY` | classification, recipient check and verification with the OpenAI Decisions API (`gpt-6-luna`, public beta); also the extraction fallback | the decisions fall back to TypeSafe JEV |
| `TYPESAFE_API_KEY` | TypeSafe JEV, the alternative decision model | with neither OpenAI nor TypeSafe, a keyword stand-in |
| `OPENBOX_ORG_API_KEY` + `OPENBOX_BACKEND_URL` | creating, dry-running and activating the rules | Apply records the calls it would make |

The recommended setup sets the first two: Claude for extraction, the
Decisions API for everything else. `JUDGE_PROVIDER=typesafe` keeps JEV as the
decision model; it measured the same accuracy. `EXTRACT_PROVIDER` and
`JUDGE_PROVIDER` override the choice made from the keys; `.env.example` lists
every setting.

## Run

```bash
uv sync
cp .env.example .env            # fill in the keys above; blank keys run that stage offline
uv run nda-compile fixtures/coca_cola_nda.md --bindings bindings/coca-cola.yaml
uv run nda-web                  # http://127.0.0.1:8010
```

Against a hosted environment, set `OPENBOX_BACKEND_URL` and an org key for
it; for example staging is `https://openbox-api.node.lat`. The org key needs
the rules API (`/agent/:id/policy-rule`), and the agent in the bindings must
exist in that org.

Any stage whose key is missing runs a deterministic fake, so the pipeline
works end to end with no credentials and the payloads it would send can be
read before an org key exists.

The verifier rarely scores a correct rule at 0.8 or above, so the CLI's
automatic apply proposes almost everything for review. In the web Studio a
person ticks the rules to apply, which is the intended flow.

## Accuracy

Measured on 30 confidentiality agreements filed with the SEC (Schedule TO
and 14D-9 exhibits), 5 runs each, against a clause-by-clause answer key;
a run's accuracy is correct rules ÷ (expected + generated − correct).

With the setup above (Claude Sonnet 5.5 extraction with the v2 prompt, the
OpenAI Decisions API with the recipient check), 90% of runs on the 20
agreements never used for tuning, and 93% across all 30, proposed exactly the
right rules.

The two agreements that differ from the answer key are judgement calls: one
where consent governs who may count as a Representative, and one where two
clauses impose overlapping duties and the stricter one sets the decision. The
answer key has not been reviewed by a lawyer.

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
