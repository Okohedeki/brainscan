"""HTTP server: OpenAI-compatible chat endpoint, recorder API, live feed and scanner UI.

Run inside WSL:   scripts/run.sh -m brainscan.server
Point any OpenAI-compatible harness at http://localhost:8765/v1 and open
http://localhost:8765/ for the scanner. Send `X-Brainscan-Run: <id>` to group
requests into a run explicitly; otherwise runs are inferred from shared prefixes.
"""

from __future__ import annotations

import os

# Less fragmentation from the very different tensor sizes the recorder and workbench allocate.
# Must be set before CUDA initialises.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import asyncio  # noqa: E402
import json
import struct
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import PROJECT_ROOT, Settings
from .recorder import workbench
from .recorder.model import Subject
from .recorder.session import GenParams, Recorder, compact
from .recorder.store import RunStore

UI_DIR = PROJECT_ROOT / "ui"
settings = Settings()


class LiveHub:
    """Fans recorder messages out to every connected scanner, batched every ~30 ms."""

    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queue: asyncio.Queue | None = None

    def publish(self, msg: dict) -> None:
        if self.loop is not None and self.queue is not None:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, msg)

    async def pump(self) -> None:
        while True:
            batch = [await self.queue.get()]
            while not self.queue.empty() and len(batch) < 400:
                batch.append(self.queue.get_nowait())
            if self.clients:
                data = json.dumps(batch)
                for ws in list(self.clients):
                    try:
                        await ws.send_text(data)
                    except Exception:
                        self.clients.discard(ws)
            await asyncio.sleep(0.03)


hub = LiveHub()
state: dict = {"recorder": None, "status": "loading", "step": "starting", "error": None}


def _load() -> None:
    try:
        state["step"] = f"loading {settings.model}"
        subject = Subject(settings.model, device=settings.device)
        rec = Recorder(subject, RunStore(settings.data_dir, settings.disk_budget_gb), settings, hub.publish)
        state["step"] = "building direction bank"
        with rec.lock:
            rec.directions.ensure_defaults()
        state.update(recorder=rec, status="ready", step="ready")
        hub.publish({"k": "status", "status": "ready"})
        print(f"[brainscan] ready: {subject.name} ({subject.n_layers} layers) on {subject.device}", flush=True)
    except Exception as exc:
        traceback.print_exc()
        state.update(status="error", error=f"{type(exc).__name__}: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub.loop, hub.queue = asyncio.get_running_loop(), asyncio.Queue()
    pump = asyncio.create_task(hub.pump())
    threading.Thread(target=_load, daemon=True).start()
    yield
    pump.cancel()


app = FastAPI(title="Brainscan recorder", lifespan=lifespan)


@app.middleware("http")
async def revalidate_ui(request: Request, call_next):
    """Make browsers revalidate the scanner files so UI updates show up on reload."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/ui/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def recorder() -> Recorder:
    if state["status"] != "ready":
        raise HTTPException(503, f"recorder {state['status']}: {state['error'] or state['step']}")
    return state["recorder"]


def _params(body: dict, request: Request) -> GenParams:
    meta = body.get("metadata") or {}
    stop = body.get("stop") or []
    tools = body.get("tools") or None
    if body.get("tool_choice") == "none":
        tools = None
    temperature = body.get("temperature")
    return GenParams(
        messages=body["messages"], tools=tools,
        temperature=0.7 if temperature is None else float(temperature),
        top_p=float(body.get("top_p") or 1.0),
        max_tokens=int(body.get("max_completion_tokens") or body.get("max_tokens") or 1024),
        stop=[stop] if isinstance(stop, str) else list(stop),
        seed=body.get("seed"),
        run_hint=request.headers.get("x-brainscan-run") or meta.get("brainscan_run"),
        label=request.headers.get("x-brainscan-label") or meta.get("brainscan_label"),
        template_kwargs={**settings.template_kwargs(), **(body.get("chat_template_kwargs") or {})})


def _usage(res) -> dict:
    return {"prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
            "total_tokens": res.prompt_tokens + res.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": res.cached_tokens}}


@app.get("/v1/models")
async def list_models():
    name = state["recorder"].s.name if state["recorder"] else settings.model
    return {"object": "list", "data": [{"id": name, "object": "model", "created": 0, "owned_by": "brainscan"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    rec = recorder()
    body = await request.json()
    params = _params(body, request)
    model_name = body.get("model") or rec.s.name
    if body.get("stream"):
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        return StreamingResponse(_stream(rec, params, model_name, include_usage), media_type="text/event-stream")
    try:
        res = await asyncio.to_thread(rec.generate, params)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    message = {"role": "assistant", "content": res.content}
    if res.reasoning:
        message["reasoning_content"] = res.reasoning
    if res.tool_calls:
        message["tool_calls"] = res.tool_calls
    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()),
        "model": model_name, "usage": _usage(res), "brainscan": {"run": res.run_id, "request": res.req},
        "choices": [{"index": 0, "message": message, "finish_reason": res.finish_reason, "logprobs": None}],
    }, headers={"X-Brainscan-Run": res.run_id})


async def _stream(rec: Recorder, params: GenParams, model_name: str, include_usage: bool):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def put(item):
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def work():
        try:
            put(("done", rec.generate(params, on_text=lambda delta: put(("text", delta)),
                                      on_reasoning=lambda delta: put(("reasoning", delta)))))
        except Exception as exc:
            traceback.print_exc()
            put(("error", exc))

    threading.Thread(target=work, daemon=True).start()
    cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())

    def chunk(delta: dict, finish: str | None = None, **extra) -> str:
        payload = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}
        return f"data: {json.dumps(payload)}\n\n"

    yield chunk({"role": "assistant", "content": ""})
    while True:
        kind, val = await queue.get()
        if kind in ("text", "reasoning"):
            yield chunk({"content": val} if kind == "text" else {"reasoning_content": val})
            continue
        if kind == "done":
            if val.tool_calls:
                yield chunk({"tool_calls": [{"index": i, **c} for i, c in enumerate(val.tool_calls)]})
            yield chunk({}, val.finish_reason, brainscan={"run": val.run_id, "request": val.req})
            if include_usage:
                payload = {"id": cid, "object": "chat.completion.chunk", "created": created,
                           "model": model_name, "choices": [], "usage": _usage(val)}
                yield f"data: {json.dumps(payload)}\n\n"
        else:
            yield f"data: {json.dumps({'error': {'message': str(val), 'type': type(val).__name__}})}\n\n"
        yield "data: [DONE]\n\n"
        return


@app.get("/brainscan/status")
async def get_status():
    out = {"status": state["status"], "step": state["step"], "error": state["error"],
           "settings": {"model": settings.model, "data_dir": str(settings.data_dir),
                        "disk_budget_gb": settings.disk_budget_gb, "max_context": settings.max_context}}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        out["gpu"] = {"name": torch.cuda.get_device_name(0), "used": total - free, "total": total}
    if state["recorder"]:
        out.update(state["recorder"].status())
    return out


@app.get("/brainscan/runs")
async def get_runs():
    return recorder().list_runs()


def _run_or_404(run_id: str):
    run = recorder().get_run(run_id)
    if run is None:
        raise HTTPException(404, f"no run {run_id}")
    return run


@app.get("/brainscan/runs/{run_id}")
async def get_run(run_id: str):
    run = _run_or_404(run_id)
    return {"meta": run.meta, "messages": run.messages, "events": run.events}


@app.get("/brainscan/runs/{run_id}/tape")
async def get_tape(run_id: str):
    return [compact(r) for r in _run_or_404(run_id).tape]


@app.get("/brainscan/runs/{run_id}/tokens/{t}")
async def get_token(run_id: str, t: int):
    detail = recorder().token_detail(run_id, t)
    if detail is None:
        raise HTTPException(404, f"no token {t} in run {run_id}")
    return detail


@app.post("/brainscan/events")
async def post_event(request: Request):
    body = await request.json()
    try:
        return recorder().external_event(body.get("run"), body["type"], body.get("data") or {}, body.get("t"))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/brainscan/interventions")
async def get_interventions():
    return [iv.describe() for iv in recorder().interventions.values()]


@app.post("/brainscan/interventions")
async def post_intervention(request: Request):
    body = await request.json()
    try:
        iv = await asyncio.to_thread(recorder().add_intervention, **body)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return iv.describe()


@app.patch("/brainscan/interventions/{iid}")
async def patch_intervention(iid: str, request: Request):
    body = await request.json()
    try:
        iv = await asyncio.to_thread(recorder().set_intervention, iid, bool(body.get("enabled", True)))
    except KeyError as exc:
        raise HTTPException(404, f"no intervention {iid}") from exc
    return iv.describe()


@app.delete("/brainscan/interventions/{iid}")
async def delete_intervention(iid: str):
    try:
        await asyncio.to_thread(recorder().remove_intervention, iid)
    except KeyError as exc:
        raise HTTPException(404, f"no intervention {iid}") from exc
    return {"deleted": iid}


@app.get("/brainscan/directions")
async def get_directions():
    return recorder().directions.describe()


@app.post("/brainscan/directions")
async def post_direction(request: Request):
    body = await request.json()
    rec = recorder()

    def build():
        with rec.lock:
            return rec.directions.build_contrast(body["name"], int(body["layer"]), body["positive"],
                                                 body["negative"], body.get("description", ""))
    try:
        await asyncio.to_thread(build)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return rec.directions.describe()


@app.post("/brainscan/directions/analyze")
async def analyze_direction(request: Request):
    body = await request.json()
    try:
        return await asyncio.to_thread(recorder().analyze_direction, body.get("positive") or [],
                                       body.get("negative") or [], body.get("position", "last"),
                                       bool(body.get("chat", True)))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/brainscan/directions/save")
async def save_direction(request: Request):
    body = await request.json()
    rec = recorder()

    def save():
        with rec.lock:
            return rec.directions.save_analysis(body["id"], int(body["layer"]), body["name"],
                                                body.get("description", ""))
    try:
        await asyncio.to_thread(save)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return rec.directions.describe()


@app.delete("/brainscan/directions/{name}")
async def delete_direction(name: str):
    rec = recorder()
    if any(e.direction == name for e in rec.edits.edits.values()) or \
            any(iv.direction == name for iv in rec.interventions.values()):
        raise HTTPException(409, f"{name} is used by an edit or intervention; remove those first")
    try:
        with rec.lock:
            rec.directions.delete(name)
    except KeyError as exc:
        raise HTTPException(404, f"no direction {name}") from exc
    return rec.directions.describe()


@app.get("/brainscan/workbench")
async def get_workbench():
    return recorder().edits.summary()


@app.post("/brainscan/workbench/edits")
async def post_weight_edit(request: Request):
    body = await request.json()
    try:
        edit = await asyncio.to_thread(recorder().add_weight_edit, **body)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return edit.describe()


@app.patch("/brainscan/workbench/edits/{eid}")
async def patch_weight_edit(eid: str, request: Request):
    body = await request.json()
    try:
        return recorder().edits.set_enabled(eid, bool(body.get("enabled", True))).describe()
    except KeyError as exc:
        raise HTTPException(404, f"no edit {eid}") from exc


@app.delete("/brainscan/workbench/edits/{eid}")
async def delete_weight_edit(eid: str):
    rec = recorder()
    if eid in rec.edits.applied:
        raise HTTPException(409, "this edit is applied; restore the original weights first")
    try:
        rec.edits.remove(eid)
    except KeyError as exc:
        raise HTTPException(404, f"no edit {eid}") from exc
    return rec.edits.summary()


@app.post("/brainscan/workbench/apply")
async def apply_weight_edits():
    try:
        return await asyncio.to_thread(recorder().apply_weight_edits)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/brainscan/workbench/revert")
async def revert_weight_edits():
    return await asyncio.to_thread(recorder().revert_weight_edits)


@app.post("/brainscan/workbench/compare")
async def compare_weight_edits(request: Request):
    body = await request.json()
    try:
        return await asyncio.to_thread(workbench.compare, recorder(), body.get("prompts") or [],
                                       int(body.get("max_tokens") or 96), body.get("reference") or None)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/brainscan/workbench/export")
async def export_edited_model(request: Request):
    body = await request.json()
    try:
        return await asyncio.to_thread(workbench.export, recorder(), body["path"])
    except (KeyError, ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


def _anatomy(rec: Recorder):
    if getattr(rec, "anatomy", None) is None:
        raise HTTPException(404, "weight anatomy is off (BRAINSCAN_ANATOMY=0)")
    return rec.anatomy


@app.get("/brainscan/anatomy")
async def get_anatomy():
    rec = recorder()
    an = _anatomy(rec)
    return an.describe() | {"edited": an.edited_units(rec.edits), "edits_version": rec.edits.version}


@app.get("/brainscan/anatomy/edited")
async def get_edited_units():
    rec = recorder()
    return {"edited": _anatomy(rec).edited_units(rec.edits), "edits_version": rec.edits.version}


@app.get("/brainscan/anatomy/edges")
async def get_edges():
    return _anatomy(recorder()).describe_edges()


@app.get("/brainscan/anatomy/params")
async def get_params():
    return _anatomy(recorder()).param_list()


@app.get("/brainscan/anatomy/units/{g}")
async def get_unit(g: int):
    rec = recorder()
    an = _anatomy(rec)
    try:
        return await asyncio.to_thread(an.unit_detail, g, set(an.edited_units(rec.edits)))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/brainscan/anatomy/units/{g}/lesion")
async def lesion_unit(g: int, request: Request):
    """Add a workbench edit that scales this unit's output weights (factor 0 by default). It changes
    the weights only when the stack is applied."""
    body = await request.json()
    rec = recorder()
    try:
        spec = _anatomy(rec).lesion(g, float(body.get("factor", 0.0)))
        edit = await asyncio.to_thread(rec.add_weight_edit, **spec)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return edit.describe()


@app.post("/brainscan/anatomy/tile")
async def post_tile(request: Request):
    body = await request.json()
    rec = recorder()
    try:
        return await asyncio.to_thread(_anatomy(rec).tile, body, rec.edits.version)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/brainscan/runs/{run_id}/anatomy")
async def get_run_anatomy(run_id: str):
    """Per-token unit activity for a run: a 4-byte little-endian header length, a JSON header
    (tape indices, unit count, unit-map key), then one byte per unit per token, row-major."""
    rec = recorder()
    an = _anatomy(rec)
    t, codes, skipped = await asyncio.to_thread(an.read_run, rec.store, run_id)
    header = json.dumps({"n": int(len(t)), "units": an.n_units, "key": an.key, "skipped": skipped,
                         "t": t.tolist()}).encode()
    return Response(struct.pack("<I", len(header)) + header + codes.tobytes(), media_type="application/octet-stream")


@app.websocket("/brainscan/live")
async def live(ws: WebSocket):
    await ws.accept()
    hub.clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        hub.clients.discard(ws)


@app.get("/")
async def index():
    return FileResponse(UI_DIR / "index.html")


app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")


def main() -> None:
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
