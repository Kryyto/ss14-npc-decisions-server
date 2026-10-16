"""Serveur REST local pour THX-01 (https://huggingface.co/doofz/THX-01).

    .venv\\Scripts\\python -m app.main            # http://localhost:8095, doc Swagger sur /docs

Endpoints
    GET  /health              état du serveur et du modèle
    GET  /v1/models           modèle chargé
    GET  /v1/presets          schémas de questions prêts à l'emploi
    POST /v1/decide           {state, questions}             -> {answers, detected, latency_ms}
    POST /v1/decide/batch     {items: [{state, questions}]}  -> {results, latency_ms}
    POST /v1/systemone        alias TypeSafe de /v1/decide
    POST /v1/systemone/batch  alias de /v1/decide/batch
Types : choice, noul, score, number, excerpt (+ "cite": true), et rules (décision dérivée d'autres réponses).
Une question "internal": true est posée au modèle mais masquée de la réponse.
"""
import argparse
import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

import thx01  # noqa: E402  (après load_dotenv pour que HF_TOKEN / HF_HOME soient pris en compte)
from thx01 import presets  # noqa: E402
from thx01.decide import decide as thx_decide  # noqa: E402

MODEL = os.environ.get("THX01_MODEL", "doofz/THX-01")
DEVICE = os.environ.get("THX01_DEVICE") or None          # None = cuda si dispo, sinon cpu
API_KEY = os.environ.get("THX01_API_KEY") or None        # vide = pas d'authentification

QUESTION_TYPES = {"choice", "noul", "score", "number", "excerpt"}

agent: Optional[thx01.Agent] = None
_lock = threading.Lock()                                  # un seul passage GPU à la fois


@asynccontextmanager
async def lifespan(_: FastAPI):
    global agent
    t0 = time.perf_counter()
    print(f"[thx01] chargement de {MODEL} ...", flush=True)
    agent = await asyncio.to_thread(thx01.load, MODEL, DEVICE)
    print(f"[thx01] prêt sur {agent.device} en {time.perf_counter() - t0:.1f} s", flush=True)
    yield


app = FastAPI(title="THX-01 local API", version=thx01.__version__, lifespan=lifespan,
              description="Décisions calibrées THX-01 (choice, noul, score, number, excerpt, cite).")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def auth(authorization: Optional[str] = Header(default=None)):
    if API_KEY and authorization != "Bearer " + API_KEY:
        raise HTTPException(401, "missing or invalid bearer token")


State = Union[str, Dict[str, Any], List[Any]]


class DecideRequest(BaseModel):
    state: State = Field(..., description="Texte, ou objet (avec un champ `document` pour number/excerpt)")
    questions: Dict[str, Dict[str, Any]]
    model: Optional[str] = None                           # accepté pour compatibilité, ignoré


class BatchItem(BaseModel):
    state: State
    questions: Dict[str, Dict[str, Any]]


class BatchRequest(BaseModel):
    items: List[BatchItem] = Field(..., min_length=1)
    model: Optional[str] = None


def _normalize(questions: Dict[str, Dict[str, Any]]):
    """Valide les questions et accepte `question` comme synonyme de `instructions` pour tous les types
    (le runtime thx01 ne l'accepte nativement que pour number/excerpt).
    Renvoie (questions posées au modèle, questions `rules`, ids `internal` à masquer)."""
    if not questions:
        raise ValueError("`questions` must contain at least one question")
    model_qs, rules, hidden = {}, {}, set()
    for qid, q in questions.items():
        t = q.get("type")
        if q.get("internal"):
            hidden.add(qid)
        if t == "rules":
            rules[qid] = q
            continue
        if t not in QUESTION_TYPES:
            raise ValueError(f"question {qid!r}: unknown type {t!r} (expected one of {sorted(QUESTION_TYPES | {'rules'})})")
        text = q.get("instructions") or q.get("question")
        if not text:
            raise ValueError(f"question {qid!r}: missing `instructions` (or `question`)")
        if t in ("choice", "score") and not q.get("criteria"):
            raise ValueError(f"question {qid!r}: type {t!r} requires `criteria`")
        if t == "score" and not isinstance(q["criteria"], list):
            raise ValueError(f"question {qid!r}: `criteria` of a score question must be an ordered list")
        model_qs[qid] = {k: v for k, v in q.items() if k != "internal"} | {"instructions": text}
    for qid, q in rules.items():
        if not isinstance(q.get("rules"), list) or "default" not in q:
            raise ValueError(f"question {qid!r}: a rules question needs a `rules` list and a `default`")
        for i, r in enumerate(q["rules"]):
            refs = r.get("if")
            refs = refs if isinstance(refs, list) else [refs]
            missing = [x for x in refs if x not in model_qs]
            if missing or "then" not in r or ("above" in r) == ("is" in r):
                raise ValueError(f"question {qid!r}: rules[{i}] needs `if` (ids of other questions), `then`, "
                                 f"and exactly one of `above` (noul) or `is` (choice); unknown ids: {missing}")
    return model_qs, rules, hidden


def _apply_rules(answers: Dict[str, Any], rules: Dict[str, Dict[str, Any]], hidden) -> Dict[str, Any]:
    """Questions `rules` : une décision dérivée des réponses du modèle, première règle vérifiée gagnante.
    `{"if": "q" | ["q1", "q2"], "above": 0.7, "then": "x"}` : P(oui) d'au moins une question noul > seuil.
    `{"if": "q", "is": "cle", "then": "x"}`                : la question choice a répondu `cle`."""
    for qid, q in rules.items():
        choice, matched = q["default"], None
        for i, r in enumerate(q["rules"]):
            refs = r["if"] if isinstance(r["if"], list) else [r["if"]]
            if "above" in r:
                hit = [x for x in refs if answers[x].get("noul", 0.0) > r["above"]]
            else:
                hit = [x for x in refs if answers[x].get("choice") == r["is"]]
            if hit:
                choice, matched = r["then"], {"rule": i, "because": hit}
                break
        signals = {x: answers[x].get("noul", answers[x].get("choice"))
                   for r in q["rules"] for x in (r["if"] if isinstance(r["if"], list) else [r["if"]])}
        answers[qid] = {"type": "rules", "choice": choice, "matched": matched, "signals": signals}
    return {k: v for k, v in answers.items() if k not in hidden}


def _lang(state):
    d = thx01.detect_language(state)
    return {"script": d["script"], "language": d["language"]}


PLAIN = {"choice", "noul", "score"}                       # types répondus par un seul passage du modèle


def _is_plain(qs):
    return all(q["type"] in PLAIN and not q.get("cite") for q in qs.values())


def _run(state, questions) -> Dict[str, Any]:
    model_qs, rules, hidden = _normalize(questions)
    with _lock:
        out = thx_decide(agent, state, model_qs)
    out["answers"] = _apply_rules(out["answers"], rules, hidden)
    out["detected"] = _lang(state)
    return out


def _run_batch(items) -> List[Dict[str, Any]]:
    """Les items qui n'ont que des questions choice/noul/score partagent un seul passage GPU batché."""
    parsed = []
    for i, it in enumerate(items):
        try:
            parsed.append(_normalize(it.questions))
        except ValueError as e:
            raise ValueError(f"items[{i}]: {e}")
    plain = [i for i, (mq, _, _) in enumerate(parsed) if _is_plain(mq)]
    results: Dict[int, Dict[str, Any]] = {}
    if plain:
        with _lock:
            outs = agent.predict_batch([items[i].state for i in plain], [parsed[i][0] for i in plain])
        for i, o in zip(plain, outs):
            results[i] = o
    for i, it in enumerate(items):
        if i not in results:
            with _lock:
                results[i] = thx_decide(agent, it.state, parsed[i][0])
    for i, it in enumerate(items):
        results[i]["answers"] = _apply_rules(results[i]["answers"], parsed[i][1], parsed[i][2])
        results[i]["detected"] = _lang(it.state)
    return [results[i] for i in range(len(items))]


async def _call(fn, *args):
    try:
        return await asyncio.to_thread(fn, *args)
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")


@app.get("/health")
def health():
    return {"status": "ok" if agent else "loading", "model": "THX-01", "version": thx01.__version__,
            "device": str(agent.device) if agent else None}


@app.get("/v1/models", dependencies=[Depends(auth)])
def models():
    return {"object": "list", "data": [{"id": "THX-01", "source": MODEL, "device": str(agent.device)}]}


@app.get("/v1/presets", dependencies=[Depends(auth)])
def get_presets():
    return {"triage": presets.triage_questions(), "email": presets.email_questions(),
            "guard": presets.guard_questions(), "moderation": presets.moderation_questions(),
            "router": presets.router_questions()}


@app.post("/v1/decide", dependencies=[Depends(auth)])
@app.post("/v1/systemone", dependencies=[Depends(auth)], include_in_schema=False)
async def decide(req: DecideRequest):
    t0 = time.perf_counter()
    out = await _call(_run, req.state, req.questions)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return out


@app.post("/v1/decide/batch", dependencies=[Depends(auth)])
@app.post("/v1/systemone/batch", dependencies=[Depends(auth)], include_in_schema=False)
async def decide_batch(req: BatchRequest):
    t0 = time.perf_counter()
    results = await _call(_run_batch, req.items)
    return {"results": results, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8095")))
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
