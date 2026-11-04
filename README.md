# private-rag-kit

**Self-hosted RAG over private documents — hybrid search, mandatory citations, refusal when the sources don't have the answer, and an eval loop so you can prove it.**

[![CI](https://github.com/walidsadfi-evolurise/private-rag-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/walidsadfi-evolurise/private-rag-kit/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

*[Version française plus bas](#version-française)*

Most RAG demos stop at "it answered something". In a company, the questions that matter are different: **does it only say what the documents say, does it show where it got it, and does it shut up when it doesn't know?** This kit is the part of my client projects (internal knowledge bases for a medical group, a law-adjacent back office, e-commerce support teams) that answers those three questions, extracted into a small, dependency-light package you can run on a €20 VPS.

```text
$ rag ingest ./docs
Indexed 3 document(s), 9 chunk(s).

$ rag ask "How fast are refunds issued?"
Refunds are issued within 10 business days after we receive the returned goods [1].

[1] Refund policy › 3. Refunds (refund-policy.md#chunk2)

$ rag ask "What is the CEO's salary?"
I can't answer that from the documents I have access to.
```

## What it does

| Stage | What happens | Why it matters |
|---|---|---|
| **Ingest** | PDF, DOCX, HTML, Markdown, TXT → text with headings preserved | Citations read *"Refund policy › 3. Refunds"*, not *"page 12"* |
| **Chunk** | Split on headings, then paragraphs, then sentences; small overlap | A chunk never straddles two sections; boundary facts stay findable |
| **Embed** | OpenAI-compatible, Ollama (local), or a hash embedder for CI | Runs fully offline if that's the requirement |
| **Store** | In-memory/npz for small corpora, **PostgreSQL + pgvector** for production | One table, HNSW + GIN indexes, lexical search in SQL |
| **Retrieve** | BM25 **and** dense vectors, fused with Reciprocal Rank Fusion | Catches exact codes/names (lexical) *and* paraphrases (dense) |
| **Answer** | Numbered sources in the prompt; every sentence must cite `[n]` | Verified after generation: a bad citation ⇒ refusal, not a guess |
| **Refuse** | Model may say `NO_ANSWER`; empty retrieval ⇒ refusal without an LLM call | Fewer hallucinations, lower cost |
| **Evaluate** | YAML eval set → hit@k, answer accuracy, citation rate, refusal recall/precision | Change the chunker or the model and *see* what moved |
| **Serve** | FastAPI with bearer-token auth: `/ask`, `/ingest`, `/documents`, `/healthz` | Drop-in for a Slack bot, a WordPress plugin, an internal tool |

## Quick start

```bash
pip install "private-rag-kit @ git+https://github.com/walidsadfi-evolurise/private-rag-kit"
# or, from a clone: pip install -e ".[dev]"

# 1. Offline smoke test (hash embedder + fake LLM: no keys, no network)
rag ingest tests/fixtures/corpus
rag ask "When are maintenance windows?"
rag eval evalsets/demo.yaml

# 2. Real local stack (Ollama)
export RAG_EMBED_PROVIDER=ollama RAG_EMBED_MODEL=nomic-embed-text
export RAG_LLM_PROVIDER=ollama   RAG_LLM_MODEL=llama3.1:8b
rag ingest ./docs && rag ask "…"

# 3. Hosted models
export RAG_LLM_PROVIDER=anthropic RAG_LLM_MODEL=claude-sonnet-4-5 RAG_ANTHROPIC_API_KEY=…
export RAG_EMBED_PROVIDER=openai  RAG_EMBED_MODEL=text-embedding-3-small RAG_OPENAI_API_KEY=…
```

All settings are environment variables prefixed `RAG_` (or a `.env` file). See [`.env.example`](.env.example).

### Production: PostgreSQL + pgvector

```bash
pip install "private-rag-kit[postgres]"
export RAG_STORE_URL=postgresql://rag:rag@localhost:5432/rag
rag ingest ./docs
```

Or with Docker:

```bash
cp .env.example .env         # set your provider + keys
docker compose -f deploy/compose.yml up -d
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/ingest -H "Authorization: Bearer $RAG_API_TOKEN" -F files=@policy.pdf
curl -s -X POST localhost:8000/ask -H "Authorization: Bearer $RAG_API_TOKEN" \
     -H 'content-type: application/json' -d '{"question":"How fast are refunds issued?"}'
```

## The evaluation loop

`evalsets/demo.yaml`:

```yaml
- question: How fast are refunds issued?
  expected_sources: [refund-policy.md]
  expected_answer_contains: ["10 business days"]
- question: What is the CEO's salary?
  expect_refusal: true
```

```text
$ rag eval evalsets/demo.yaml --min-accuracy 0.8
 metric            value
 hit_at_k           100%   ← a relevant chunk was in the top-k
 answer_accuracy    100%   ← answer contains the expected facts
 citation_rate      100%   ← every cited passage is a relevant source
 refusal_recall     100%   ← unanswerable questions were refused
 refusal_precision  100%   ← refusals were only on unanswerable questions
```

`--min-accuracy` makes it a CI gate. Write 20–50 real questions from your users' tickets before you tune anything; it's the only way to know whether a change to chunking, embeddings or the prompt helped.

## Design notes

- **RRF instead of score interpolation.** BM25 and cosine scores live on different scales; rank fusion needs no tuning and is robust across corpora.
- **Citations are verified, not trusted.** `verify_citations` checks that every sentence carries a marker and every marker points at a retrieved source. Fail either and the answer is downgraded to the refusal text. The model's own `NO_ANSWER` is always believed.
- **Refusal without an LLM call** when retrieval returns nothing: cheaper, and it never gets the chance to improvise.
- **Hash embedder on purpose.** It's semantically weak, but deterministic and offline, so the test suite and CI exercise the whole pipeline — store, fusion, prompt, verification, API — in under a second with no keys. Never use it for real answers.
- **Minimal surface.** No framework-of-frameworks. ~1 400 lines, typed (`mypy --strict`), 27 tests, `ruff`.

## Project layout

```
src/ragkit/
  ingest/      extract.py (pdf/docx/html/md) · chunk.py (heading-aware)
  embed/       hashing.py · http.py (OpenAI-compatible, Ollama)
  store/       memory.py · pgvector.py
  llm/         base.py (FakeLLM) · http.py (Anthropic, OpenAI, Ollama)
  retrieval.py hybrid BM25 + vectors, RRF
  answer.py    prompt, citation verification, refusal
  evaluate.py  metrics
  cli.py       rag ingest | ask | eval | status | serve
  api.py       FastAPI
deploy/        compose.yml (api + postgres/pgvector), Dockerfile
evalsets/      demo.yaml
```

## Roadmap

- [ ] Reranker step (cross-encoder / LLM rerank) behind a flag
- [ ] Per-document ACL: filter chunks by caller group before retrieval
- [ ] Incremental ingest (skip unchanged files by hash — `doc_id` is already content-addressed)
- [ ] Streaming `/ask`

## License

MIT — see [LICENSE](LICENSE). Built by [Walid Sadfi](https://www.malt.fr/profile/walidsadfi), Forward Deployed Engineer · agents IA & WooCommerce.

---

## Version française

**RAG auto-hébergé sur documents privés : recherche hybride, citations obligatoires, refus quand les sources ne contiennent pas la réponse, et une boucle d'évaluation pour le prouver.**

La plupart des démos RAG s'arrêtent à « ça a répondu quelque chose ». En entreprise, les vraies questions sont : **est-ce qu'il ne dit que ce que disent les documents, est-ce qu'il montre d'où ça vient, et est-ce qu'il se tait quand il ne sait pas ?** Ce kit est la partie de mes projets clients (base de connaissances interne d'un groupe médical, back-office, support e-commerce) qui répond à ces trois questions, extraite en un petit package que vous pouvez faire tourner sur un VPS à 20 €.

- **Ingestion** PDF / DOCX / HTML / Markdown avec conservation des titres → citations lisibles (*« Politique de remboursement › 3. Remboursements »*).
- **Recherche hybride** BM25 + vecteurs, fusion RRF : trouve les codes et noms exacts *et* les paraphrases.
- **Citations vérifiées** après génération : une citation fausse ⇒ refus, jamais une invention.
- **Refus sans appel LLM** quand la recherche ne retourne rien.
- **Évaluation** : jeu de questions YAML → hit@k, exactitude, taux de citation, rappel/précision du refus. `--min-accuracy` en fait une barrière CI.
- **Déploiement** : `docker compose` avec PostgreSQL + pgvector, API FastAPI protégée par token.
- **Local ou hébergé** : Ollama en local pour les données sensibles, Anthropic/OpenAI sinon.

```bash
pip install -e ".[dev]"
rag ingest tests/fixtures/corpus
rag ask "Quand ont lieu les fenêtres de maintenance ?"
rag eval evalsets/demo.yaml
```

Besoin de ça branché sur vos documents, votre Slack ou votre WordPress, en production, avec un point de mesure ? [Parlons-en sur Malt](https://www.malt.fr/profile/walidsadfi).
