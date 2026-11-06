"""Render/Playwright port of the AcidFirm visual bot.

Telegram login is completed by the owner through the protected control page.
The Chromium profile and lead calibration must live on persistent storage.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from playwright.async_api import async_playwright

VERSION = "render-v1"
TARGET_RUN_SCORE = 1000
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/acidauto-data"))
PROFILE_DIR = DATA_DIR / "telegram-profile"
LEAD_FILE = DATA_DIR / "acidfirm_lead.json"
TELEGRAM_WEB_URL = os.environ.get("TELEGRAM_WEB_URL", "https://web.telegram.org/a/#@acidfirm_bot")
VIEW_W = int(os.environ.get("VIEW_W", "523"))
VIEW_H = int(os.environ.get("VIEW_H", "925"))
try:
    INITIAL_LEAD_MS = float(os.environ.get("INITIAL_LEAD_MS", "58.5"))
except ValueError:
    INITIAL_LEAD_MS = 58.5
INITIAL_LEAD_MS = INITIAL_LEAD_MS if 45.0 <= INITIAL_LEAD_MS <= 90.0 else 58.5

app = FastAPI(title="AcidFirm private runner", docs_url=None, redoc_url=None)
pw = None
context = None
page = None
bot_task = None
commands: asyncio.Queue = asyncio.Queue()
latest_frame = b""
runtime = {"running": False, "browser_ready": False, "score": 0, "drops": 0, "ideal": 0, "lead_ms": INITIAL_LEAD_MS, "error": "starting"}


def extract_bar_bounds(val, r_top, r_bot, x1, y1, min_bar_w, max_bar_h):
    """
    Berilgan vertikal qatorlar oralig'idan bitta blokning chap/o'ng chegarasini topish.
    O'rtadagi ball raqami (score digit) va qo'shni bloklardan to'liq himoyalangan.
    """
    sub_bot = min(r_bot, r_top + max_bar_h)
    seg_h = sub_bot - r_top + 1
    if seg_h < 8:
        return None

    pad = max(2, seg_h // 6)
    mid_r1 = r_top + pad
    mid_r2 = max(mid_r1 + 2, sub_bot - pad)

    # To'q rangli bloklarda eski chegara juda qattiq bo'lib, blokni o'tkazib yuborardi.
    col_mask = (val[mid_r1:mid_r2] > 8).mean(axis=0) >= 0.50
    xs = np.where(col_mask)[0]
    if len(xs) < min_bar_w:
        return None

    gaps = np.where(np.diff(xs) > 3)[0]
    if len(gaps) > 0:
        starts = np.r_[0, gaps + 1]
        ends = np.r_[gaps, len(xs) - 1]
        best_idx = int(np.argmax(xs[ends] - xs[starts]))
        left_rel = xs[starts[best_idx]]
        right_rel = xs[ends[best_idx]]
    else:
        left_rel = xs[0]
        right_rel = xs[-1]

    width = int(right_rel - left_rel + 1)
    if width < min_bar_w:
        return None
    # To'liq ekran kengligidagi fon chizig'i blok sifatida qabul qilinmasin.
    if width > int(val.shape[1] * 0.85):
        return None

    block_patch = val[r_top:sub_bot + 1, left_rel:right_rel + 1]
    if block_patch.max() < 32:
        return None
    top_edge_patch = val[r_top:min(sub_bot + 1, r_top + 5), left_rel:right_rel + 1]
    if (top_edge_patch > 26).max(axis=0).sum() < max(5, int(width * 0.20)):
        return None

    left_x = int(left_rel + x1)
    right_x = int(right_rel + x1)
    return {
        "y_top": int(r_top + y1),
        "y_bot": int(sub_bot + y1),
        "left": left_x,
        "right": right_x,
        "width": width,
        "center": (left_x + right_x) / 2.0,
    }


def detect_bars(frame_bgra):
    """
    O'yin kadridan:
      1) Tepada harakatlanayotgan blokni (moving_bar)
      2) Pastda turgan eng ustki mo'ljal blokni (target_bar)
    aniqlaydi.
    """
    H, W = frame_bgra.shape[:2]
    y1, y2 = int(H * 0.455), int(H * 0.90)
    x1, x2 = 4, max(10, W - 4)

    val = frame_bgra[y1:y2, x1:x2, :3].max(axis=2)

    min_bar_w = max(10, int((x2 - x1) * 0.025))
    min_bar_h = max(9, int(H * 0.018))
    max_bar_h = max(15, int(H * 0.035))

    # Yulduzcha fonidagi alohida nuqtalar qatorni to'ldirib ko'rsatmasin:
    # kamida bir necha piksel davom etgan gorizontal izlarni qoldiramiz.
    row_mask = (val > 9).astype(np.uint8)
    line_width = max(6, int((x2 - x1) * 0.02))
    horizontal_kernel = np.ones((1, line_width), dtype=np.uint8)
    horizontal_mask = cv2.morphologyEx(row_mask, cv2.MORPH_OPEN, horizontal_kernel)
    row_counts = horizontal_mask.sum(axis=1)
    row_body = row_counts >= max(min_bar_w, int((x2 - x1) * 0.04))
    row_valid_bright = (val > 24).sum(axis=1) >= max(3, min_bar_w // 4)

    raw_segments = []
    in_seg = False
    start_r = 0
    for r in range(len(row_body)):
        if row_body[r] and not in_seg:
            in_seg = True
            start_r = r
        elif not row_body[r] and in_seg:
            in_seg = False
            if r - start_r >= min_bar_h and np.any(row_valid_bright[start_r:r]):
                raw_segments.append((start_r, r - 1))
    if in_seg and len(row_body) - start_r >= min_bar_h and np.any(row_valid_bright[start_r:]):
        raw_segments.append((start_r, len(row_body) - 1))

    if not raw_segments:
        return None, None

    bars = []
    for idx, (r_top, r_bot) in enumerate(raw_segments):
        seg_h = r_bot - r_top + 1
        if idx == 0 and seg_h >= int(max_bar_h * 1.60):
            s1 = min_bar_h
            s2 = min(seg_h - min_bar_h, int(max_bar_h * 1.30))
            if s2 > s1 + 2:
                gap_profile = (val[r_top + s1:r_top + s2] > 24).sum(axis=1)
                split_offset = int(np.argmin(gap_profile)) + s1
                b1 = extract_bar_bounds(val, r_top, r_top + split_offset - 1, x1, y1, min_bar_w, max_bar_h)
                b2 = extract_bar_bounds(val, r_top + split_offset + 1, r_bot, x1, y1, min_bar_w, max_bar_h)
                if b1:
                    bars.append(b1)
                if b2:
                    bars.append(b2)
                continue

        b = extract_bar_bounds(val, r_top, r_bot, x1, y1, min_bar_w, max_bar_h)
        if b:
            bars.append(b)

    for i in range(len(bars) - 1):
        b_top = bars[i]
        b_bot = bars[i + 1]
        v_gap = b_bot["y_top"] - b_top["y_bot"]
        if -2 <= v_gap <= int(max_bar_h * 0.85):
            return b_top, b_bot

    return None, None


def save_debug_image(frame_bgra, moving_bar, target_bar, filename="debug_frame.png"):
    """Bot bloklarni qanday ko'rayotganini rasmga chizib saqlash."""
    vis = frame_bgra[:, :, :3].copy()
    if moving_bar:
        cv2.rectangle(
            vis,
            (moving_bar["left"], moving_bar["y_top"]),
            (moving_bar["right"], moving_bar["y_bot"]),
            (0, 255, 255),
            2,
        )
        cx = int(moving_bar["center"])
        cv2.line(vis, (cx, moving_bar["y_top"] - 10), (cx, moving_bar["y_bot"] + 10), (0, 255, 255), 2)
    if target_bar:
        cv2.rectangle(
            vis,
            (target_bar["left"], target_bar["y_top"]),
            (target_bar["right"], target_bar["y_bot"]),
            (0, 255, 0),
            2,
        )
        tx = int(target_bar["center"])
        cv2.line(vis, (tx, target_bar["y_top"] - 35), (tx, target_bar["y_bot"] + 10), (0, 255, 0), 2)
    cv2.imwrite(filename, vis)
    return filename


class Command(BaseModel):
    action: str
    x: Optional[float] = None
    y: Optional[float] = None


def require_token(token: Optional[str]):
    if not ADMIN_TOKEN or not token or not hmac.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Bad admin token")


def load_lead() -> float:
    try:
        value = float(json.loads(LEAD_FILE.read_text("utf-8")).get("lead_time_ms", INITIAL_LEAD_MS))
        return value if 45.0 <= value <= 90.0 else INITIAL_LEAD_MS
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return INITIAL_LEAD_MS


def save_lead(value: float) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LEAD_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"lead_time_ms": round(value, 2)}), encoding="utf-8")
    os.replace(tmp, LEAD_FILE)


@app.on_event("startup")
async def startup():
    global pw, context, page, bot_task
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    if not ADMIN_TOKEN:
        runtime["error"] = "Set ADMIN_TOKEN in Render environment variables"
        raise RuntimeError(runtime["error"])
    pw = await async_playwright().start()
    context = await pw.chromium.launch_persistent_context(
        str(PROFILE_DIR),
        headless=True,
        viewport={"width": VIEW_W, "height": VIEW_H},
        device_scale_factor=1,
        args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-background-timer-throttling", "--disable-renderer-backgrounding"],
    )
    page = context.pages[0] if context.pages else await context.new_page()
    await page.goto(TELEGRAM_WEB_URL, wait_until="domcontentloaded", timeout=90000)
    runtime["browser_ready"] = True
    runtime["lead_ms"] = load_lead()
    runtime["error"] = "Open Telegram or scan the login QR in the control screen"
    bot_task = asyncio.create_task(run_loop())


@app.on_event("shutdown")
async def shutdown():
    global context, pw
    runtime["running"] = False
    if bot_task:
        bot_task.cancel()
    if context:
        await context.close()
    if pw:
        await pw.stop()


@app.get("/health")
async def health():
    return {"ok": True, "version": VERSION}


@app.get("/", response_class=HTMLResponse)
async def control_page():
    return HTMLResponse(CONTROL_HTML)


@app.get("/status")
async def status(x_admin_token: Optional[str] = Header(default=None)):
    require_token(x_admin_token)
    return JSONResponse(dict(runtime))


@app.get("/screen")
async def screen(x_admin_token: Optional[str] = Header(default=None)):
    require_token(x_admin_token)
    if not latest_frame:
        raise HTTPException(status_code=503, detail="Browser frame is not ready")
    return Response(latest_frame, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.post("/command")
async def command(body: Command, x_admin_token: Optional[str] = Header(default=None)):
    require_token(x_admin_token)
    if body.action not in {"start", "pause", "click", "refresh"}:
        raise HTTPException(status_code=400, detail="Unknown action")
    if body.action == "click" and (body.x is None or body.y is None):
        raise HTTPException(status_code=400, detail="Click needs x and y")
    await commands.put(body.model_dump())
    return {"ok": True}


async def run_loop():
    global latest_frame
    history = deque(maxlen=8)
    target_centers = deque(maxlen=8)
    target_widths = deque(maxlen=8)
    target_lefts = deque(maxlen=8)
    target_rights = deque(maxlen=8)
    lead_history = deque(maxlen=5)
    timing_errors = deque(maxlen=5)
    lead = load_lead()
    runtime["lead_ms"] = lead
    last_known_speed = None
    last_drop_info = None
    locked_target = None
    drops = ideal_count = score = 0
    last_click_time = 0.0
    last_status = 0.0
    no_bars_since = None
    LEAD_GAIN = 0.10

    while True:
        try:
            while not commands.empty():
                cmd = commands.get_nowait()
                if cmd["action"] == "start":
                    runtime["running"] = True
                    runtime["error"] = ""
                elif cmd["action"] == "pause":
                    runtime["running"] = False
                elif cmd["action"] == "refresh":
                    await page.reload(wait_until="domcontentloaded", timeout=60000)
                elif cmd["action"] == "click":
                    await page.mouse.click(float(cmd["x"]), float(cmd["y"]))

            shot = await page.screenshot(type="png", timeout=15000)
            latest_frame = shot
            img = cv2.imdecode(np.frombuffer(shot, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                await asyncio.sleep(0.04)
                continue
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            runtime["browser_ready"] = page is not None and not page.is_closed()

            if not runtime["running"]:
                await asyncio.sleep(0.18)
                continue

            if time.perf_counter() - last_click_time < 0.135:
                await asyncio.sleep(0.002)
                continue

            moving_bar, target_bar = detect_bars(frame)
            now = time.perf_counter()
            if not moving_bar or not target_bar:
                if no_bars_since is None:
                    no_bars_since = now
                history.clear()
                if now - last_click_time > 1.5:
                    locked_target = None
                    last_known_speed = None
                await asyncio.sleep(0.01)
                continue

            long_gap = no_bars_since is not None and now - no_bars_since >= 1.0
            no_bars_since = None
            if long_gap and target_bar["width"] >= 290 and drops > 0:
                locked_target = last_drop_info = None
                last_known_speed = None
                history.clear()
                target_centers.clear(); target_widths.clear(); target_lefts.clear(); target_rights.clear()
                drops = ideal_count = score = 0
                runtime.update({"drops": 0, "ideal": 0, "score": 0})

            if locked_target is not None and (
                target_bar["width"] > locked_target["width"] + 4
                or target_bar["left"] < locked_target["left"] - 5
                or target_bar["right"] > locked_target["right"] + 5
            ):
                await asyncio.sleep(0.002)
                continue

            target_centers.append(target_bar["center"])
            target_widths.append(target_bar["width"])
            target_lefts.append(target_bar["left"])
            target_rights.append(target_bar["right"])
            tc = float(np.median(target_centers)); tw = int(round(np.median(target_widths)))
            tl = int(round(np.median(target_lefts))); tr = int(round(np.median(target_rights)))
            if len(target_centers) >= 2:
                locked_target = {"center": tc, "width": tw, "left": tl, "right": tr}

            # Score each placement after the next settled target is visible.
            if last_drop_info is not None and not last_drop_info["calibrated"] and len(target_centers) >= 3:
                last_drop_info["calibrated"] = True
                score += 1
                w_loss = last_drop_info["width"] - tw
                shift = tc - last_drop_info["center"]
                if 0 <= w_loss <= 60 and abs(shift) <= 35:
                    if w_loss <= 1 and abs(shift) <= 1.5:
                        ideal_count += 1; score += 1
                    else:
                        err_px = 2.0 * shift
                        vx_before = last_drop_info["vx"]
                        if vx_before and abs(abs(err_px) - w_loss) <= 4.5:
                            timing_errors.append((err_px / vx_before) * 1000.0)
                            if len(timing_errors) >= 3 and max(abs(x) for x in timing_errors) <= 35:
                                filt = float(np.median(timing_errors))
                                lead = float(np.clip(lead + filt * LEAD_GAIN, 45.0, 90.0))
                                lead_history.append(lead)
                                if len(lead_history) >= 3:
                                    save_lead(float(np.median(lead_history)))
                                runtime["lead_ms"] = round(lead, 1)
                runtime.update({"score": score, "ideal": ideal_count, "drops": drops})
                if score >= TARGET_RUN_SCORE:
                    runtime["running"] = False
                    runtime["error"] = "1000 estimate reached; paused"
                    continue

            m = float(moving_bar["center"])
            if history:
                pt, px = history[-1]
                if abs(m - px) > 85 or now - pt > 0.22:
                    history.clear()
                elif len(history) >= 2:
                    if (history[-1][1] - history[-2][1]) * (m - px) < 0:
                        history.clear()
            if not history or abs(m - history[-1][1]) >= 0.5:
                history.append((now, m))
            if len(history) < 2:
                await asyncio.sleep(0.002); continue

            ts = np.array([v[0] for v in history], dtype=np.float64)
            xs = np.array([v[1] for v in history], dtype=np.float64)
            span = ts[-1] - ts[0]
            if span <= 0.006:
                await asyncio.sleep(0.002); continue
            tref = ts[-1]
            if len(history) >= 3:
                vx, xref = np.polyfit(ts - tref, xs, 1)
                if last_known_speed and abs(abs(vx) - last_known_speed) / last_known_speed < 0.25:
                    expected = min(900.0, last_known_speed * 1.012)
                    clamp = float(np.clip(abs(vx), expected * .94, expected * 1.06))
                    vx = float(np.sign(vx) * (.60 * clamp + .40 * expected))
            else:
                raw = (xs[-1] - xs[0]) / span
                if last_known_speed and abs(abs(raw) - last_known_speed) / last_known_speed < .30:
                    expected = min(900.0, last_known_speed * 1.012)
                    vx = float(np.sign(raw) * (.45 * abs(raw) + .55 * expected))
                    xref = float(xs[-1])
                else:
                    await asyncio.sleep(.002); continue

            if not 200.0 <= abs(vx) <= 900.0 or (last_known_speed and abs(vx) > last_known_speed * 1.30):
                await asyncio.sleep(.002); continue
            if (tc - xref) * vx < -12.0:
                await asyncio.sleep(.002); continue
            tnow = time.perf_counter()
            xnow = xref + vx * (tnow - tref)
            eta = ((tc - xnow) / vx) - lead / 1000.0
            if eta < -0.002:
                await asyncio.sleep(.002); continue
            if eta <= 0.058:
                if eta > 0:
                    await asyncio.sleep(eta)
                click_x, click_y = VIEW_W / 2, VIEW_H * .38
                click_time = time.perf_counter()
                await page.mouse.click(click_x, click_y)
                last_click_time = time.perf_counter()
                drops += 1
                measured = min(900.0, abs(float(vx)))
                last_known_speed = measured if last_known_speed is None else .70 * last_known_speed + .30 * measured
                last_drop_info = {"center": tc, "width": tw, "vx": float(vx), "calibrated": False}
                runtime.update({"drops": drops, "ideal": ideal_count, "score": score, "lead_ms": round(lead, 1), "error": ""})
                history.clear(); target_centers.clear(); target_widths.clear(); target_lefts.clear(); target_rights.clear()
            if now - last_status > .25:
                last_status = now
            await asyncio.sleep(.001)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            runtime["running"] = False
            runtime["error"] = f"runtime: {type(exc).__name__}: {exc}"
            await asyncio.sleep(1.0)


CONTROL_HTML = r'''<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><title>AcidFirm control</title>
<style>body{font:16px system-ui;background:#111;color:#eee;margin:16px}input,button{padding:10px;margin:4px}img{max-width:min(100%,523px);height:auto;display:block;margin-top:12px;touch-action:none}#state{white-space:pre-wrap}</style>
<h2>AcidFirm server</h2><input id="token" type="password" placeholder="Render ADMIN_TOKEN"><button onclick="connect()">Connect</button>
<button onclick="cmd('refresh')">Refresh Telegram</button><button onclick="cmd('start')">Start bot</button><button onclick="cmd('pause')">Pause</button>
<div id="state">Enter the admin token.</div><img id="screen" alt="Telegram Web screen; tap it to navigate/login">
<script>
let key=sessionStorage.getItem('acid_token')||''; document.querySelector('#token').value=key;
function headers(){return {'X-Admin-Token':key,'Content-Type':'application/json'}}
async function connect(){key=document.querySelector('#token').value;sessionStorage.setItem('acid_token',key);await refresh()}
async function cmd(action){await fetch('/command',{method:'POST',headers:headers(),body:JSON.stringify({action})});}
async function refresh(){try{let s=await fetch('/status',{headers:headers()});if(!s.ok)throw Error('Token invalid');document.querySelector('#state').textContent=JSON.stringify(await s.json(),null,2);let f=await fetch('/screen',{headers:headers()});if(f.ok)document.querySelector('#screen').src=URL.createObjectURL(await f.blob())}catch(e){document.querySelector('#state').textContent=e.message}}
document.querySelector('#screen').addEventListener('pointerdown',async e=>{let r=e.target.getBoundingClientRect();let x=(e.clientX-r.left)*e.target.naturalWidth/r.width;let y=(e.clientY-r.top)*e.target.naturalHeight/r.height;await fetch('/command',{method:'POST',headers:headers(),body:JSON.stringify({action:'click',x,y})})});
setInterval(()=>{if(key)refresh()},700); if(key)connect();
</script>'''


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))

