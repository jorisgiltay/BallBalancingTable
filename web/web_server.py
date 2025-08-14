import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

from aiohttp import web, WSMsgType


class WebInterface:
    """
    Lightweight aiohttp-based web UI with WebSocket streaming for telemetry and
    command channel for live parameter tuning. Runs in a dedicated thread with
    its own asyncio loop so the control loop is never blocked.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765,
                 static_dir: Optional[Path] = None,
                 broadcast_hz: float = 20.0):
        self.host = host
        self.port = port
        self.static_dir = static_dir or (Path(__file__).parent)
        self.broadcast_period = 1.0 / max(1.0, float(broadcast_hz))

        # Runtime state
        self._controller: Any = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._clients: Set[web.WebSocketResponse] = set()

        # Latest telemetry payload; stored as JSON string to avoid re-serialization cost
        self._latest_payload_json: Optional[str] = None
        self._latest_lock = threading.Lock()
        self._running = threading.Event()
        self._new_data_event: Optional[asyncio.Event] = None
        self._last_snapshot_t: float = 0.0

    def attach_controller(self, controller: Any) -> None:
        self._controller = controller

    def publish_sample(self, payload: Dict[str, Any]) -> None:
        """Fast, non-blocking update from control thread. Payload must be JSON-serializable.
        Minimizes overhead by pre-serializing to string once.
        """
        try:
            # Only include necessary fields for the UI; coerce numpy types to plain python
            def as_float(v, default=0.0):
                try:
                    return float(v)
                except Exception:
                    return float(default)
            def as_list_of_float(seq, length=2):
                try:
                    return [as_float(seq[i], 0.0) for i in range(length)]
                except Exception:
                    try:
                        return [as_float(x, 0.0) for x in seq]
                    except Exception:
                        return [0.0, 0.0]

            imu_in = payload.get("imu", {"connected": False}) or {"connected": False}
            imu_payload = {
                "connected": bool(imu_in.get("connected", False)),
                "pitch": as_float(imu_in.get("pitch", 0.0), 0.0),
                "roll": as_float(imu_in.get("roll", 0.0), 0.0),
                "heading": as_float(imu_in.get("heading", 0.0), 0.0),
            }

            payload_small = {
                "t": as_float(payload.get("t", time.time()), time.time()),
                "step": int(payload.get("step", 0) or 0),
                "method": payload.get("method"),
                "ball_x": as_float(payload.get("ball_x", 0.0), 0.0),
                "ball_y": as_float(payload.get("ball_y", 0.0), 0.0),
                "setpoint_x": as_float(payload.get("setpoint_x", 0.0), 0.0),
                "setpoint_y": as_float(payload.get("setpoint_y", 0.0), 0.0),
                "pitch": as_float(payload.get("pitch", 0.0), 0.0),
                "roll": as_float(payload.get("roll", 0.0), 0.0),
                "action": as_list_of_float(payload.get("action", [0.0, 0.0]), 2),
                "imu": imu_payload,
            }
            encoded = json.dumps({"type": "telemetry", "data": payload_small})
            with self._latest_lock:
                self._latest_payload_json = encoded
            # Nudge the broadcaster for timely delivery (non-blocking)
            if self._loop and self._new_data_event is not None:
                try:
                    self._loop.call_soon_threadsafe(self._new_data_event.set)
                except Exception:
                    pass
        except Exception:
            # Never raise into control loop
            pass

    # Public lifecycle
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run_loop_thread, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._shutdown_async(), self._loop)
        if self._thread:
            self._thread.join(timeout=1.5)

    # Internal: thread target
    def _run_loop_thread(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._app = web.Application()
        self._app.add_routes([
            web.get("/", self._handle_index),
            web.get("/ws", self._handle_ws),
            web.static("/static", str(self.static_dir))
        ])

        self._new_data_event = asyncio.Event()
        self._loop.create_task(self._startup_async())
        try:
            self._loop.run_until_complete(self._broadcast_loop())
        finally:
            try:
                self._loop.run_until_complete(self._shutdown_async())
            except Exception:
                pass
            self._loop.close()

    async def _startup_async(self) -> None:
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

    async def _shutdown_async(self) -> None:
        try:
            for ws in list(self._clients):
                try:
                    await ws.close()
                except Exception:
                    pass
        finally:
            self._clients.clear()
            if self._runner:
                try:
                    await self._runner.cleanup()
                except Exception:
                    pass

    # HTTP handlers
    async def _handle_index(self, request: web.Request) -> web.Response:
        index_path = self.static_dir / "index.html"
        if index_path.exists():
            return web.FileResponse(str(index_path))
        # Fallback minimal page
        html = """
<!doctype html>
<html>
  <head><meta charset=\"utf-8\"><title>Ball Balance UI</title></head>
  <body>
    <h3>Ball Balance UI</h3>
    <p>Static files not found. Ensure web/index.html exists.</p>
  </body>
</html>
"""
        return web.Response(text=html, content_type="text/html")

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        self._clients.add(ws)
        try:
            # Send initial state
            await self._send_initial_state(ws)
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._handle_message(ws, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            self._clients.discard(ws)
        return ws

    async def _send_initial_state(self, ws: web.WebSocketResponse) -> None:
        try:
            state = self._snapshot_state()
            await ws.send_str(json.dumps({"type": "state", "data": state}))
        except Exception:
            pass

    def _snapshot_state(self) -> Dict[str, Any]:
        ctrl = self._controller
        if not ctrl:
            return {}
        try:
            return {
                "control_method": getattr(ctrl, "control_method", "pid"),
                "setpoint_x": getattr(ctrl, "setpoint_x", 0.0),
                "setpoint_y": getattr(ctrl, "setpoint_y", 0.0),
                "imu_feedback_gain": getattr(ctrl, "imu_feedback_gain", 0.0),
                "pid": {
                    "pitch": {
                        "kp": getattr(getattr(ctrl, "pitch_pid", None), "kp", 0.0),
                        "ki": getattr(getattr(ctrl, "pitch_pid", None), "ki", 0.0),
                        "kd": getattr(getattr(ctrl, "pitch_pid", None), "kd", 0.0),
                    },
                    "roll": {
                        "kp": getattr(getattr(ctrl, "roll_pid", None), "kp", 0.0),
                        "ki": getattr(getattr(ctrl, "roll_pid", None), "ki", 0.0),
                        "kd": getattr(getattr(ctrl, "roll_pid", None), "kd", 0.0),
                    },
                },
            }
        except Exception:
            return {}

    async def _handle_message(self, ws: web.WebSocketResponse, data: str) -> None:
        try:
            msg = json.loads(data)
        except Exception:
            return
        if not isinstance(msg, dict):
            return

        mtype = msg.get("type")
        payload = msg.get("data", {})
        ctrl = self._controller
        if not ctrl:
            return

        try:
            if mtype == "set_setpoint":
                sx = float(payload.get("x", 0.0))
                sy = float(payload.get("y", 0.0))
                ctrl.set_setpoint(sx, sy)
            elif mtype == "update_pid":
                axis = payload.get("axis")  # "pitch" or "roll"
                kp = payload.get("kp")
                ki = payload.get("ki")
                kd = payload.get("kd")
                pid_obj = ctrl.pitch_pid if axis == "pitch" else ctrl.roll_pid
                lock = getattr(ctrl, "param_lock", None)
                if lock:
                    with lock:
                        if kp is not None:
                            pid_obj.kp = float(kp)
                        if ki is not None:
                            pid_obj.ki = float(ki)
                        if kd is not None:
                            pid_obj.kd = float(kd)
                else:
                    if kp is not None:
                        pid_obj.kp = float(kp)
                    if ki is not None:
                        pid_obj.ki = float(ki)
                    if kd is not None:
                        pid_obj.kd = float(kd)
                # Echo updated state
                await ws.send_str(json.dumps({"type": "state", "data": self._snapshot_state()}))
            elif mtype == "set_method":
                method = payload.get("method", "pid")
                if method in ("pid", "rl", "lqr"):
                    ctrl.control_method = method
                    await ws.send_str(json.dumps({"type": "state", "data": self._snapshot_state()}))
            elif mtype == "set_imu_gain":
                try:
                    gain = float(payload.get("gain", 0.0))
                except Exception:
                    gain = 0.0
                gain = max(0.0, min(1.0, gain))
                lock = getattr(ctrl, "param_lock", None)
                if lock:
                    with lock:
                        setattr(ctrl, "imu_feedback_gain", gain)
                else:
                    setattr(ctrl, "imu_feedback_gain", gain)
                await ws.send_str(json.dumps({"type": "state", "data": self._snapshot_state()}))
            elif mtype == "circle":
                # Control circle mode and parameters
                on = payload.get("on")
                radius = payload.get("radius")
                speed = payload.get("speed")
                invert = payload.get("invert")
                if on is not None:
                    ctrl._circle_mode = bool(on)
                if radius is not None:
                    ctrl.set_circle_radius(float(radius))
                if speed is not None:
                    ctrl._circle_speed = float(speed)
                if invert is not None:
                    ctrl._invert_circle_direction = bool(invert)
            # Ignore unknown types silently
        except Exception:
            # Never disrupt the UI
            pass

    async def _broadcast_loop(self) -> None:
        # Ensure startup
        while self._runner is None and self._running.is_set():
            await asyncio.sleep(0.05)
        last_sent = 0.0
        while self._running.is_set():
            triggered = False
            if self._new_data_event is not None:
                try:
                    await asyncio.wait_for(self._new_data_event.wait(), timeout=self.broadcast_period)
                    triggered = True
                except asyncio.TimeoutError:
                    triggered = False
                finally:
                    if self._new_data_event.is_set():
                        self._new_data_event.clear()

            now = time.time()
            should_send = (now - last_sent) >= self.broadcast_period or (triggered and (now - last_sent) >= (self.broadcast_period * 0.5))
            if should_send:
                payload_json = None
                with self._latest_lock:
                    payload_json = self._latest_payload_json
                if payload_json and self._clients:
                    tasks = [ws.send_str(payload_json) for ws in list(self._clients)]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    # Drop failed clients
                    for ws, res in zip(list(self._clients), results):
                        if isinstance(res, Exception):
                            try:
                                await ws.close()
                            except Exception:
                                pass
                            self._clients.discard(ws)
                last_sent = now
            await asyncio.sleep(0)


