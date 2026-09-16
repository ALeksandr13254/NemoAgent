"""Computer-control tools executed on the user's machine (Windows / Linux / macOS).

Everything the model can do to this computer goes through `execute(name, args)`; dangerous
commands are gated by a confirmation callback (see TOOL_CONFIRM in .env).
"""
from __future__ import annotations

import base64
import getpass
import io
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .config import settings

log = logging.getLogger("exec")

IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

_DANGEROUS = re.compile(
    r"(\brm\b[^\n|;&]*\s-\w*[rR]|\brmdir\b|\bdel\b|\berase\b|Remove-Item|\bformat\b|\bdiskpart\b|\bmkfs|\bdd\s+if=|"
    r"\bshutdown\b|\breboot\b|Restart-Computer|Stop-Computer|\breg\s+(delete|add)\b|\bnet\s+user\b|\bnetsh\b|"
    r"Set-ExecutionPolicy|\bschtasks\b|\btakeown\b|\bicacls\b|\bcipher\b\s*/w|\bbcdedit\b|\bsfc\b|\bDISM\b|"
    r">\s*/dev/sd|\bchmod\s+-R\b|\bchown\s+-R\b|\bsudo\b|\bkill(all)?\b|Stop-Process|taskkill|\bgit\s+push\s+--force|"
    r"\bcurl\b[^\n]*\|\s*(sh|bash|powershell)|\biwr\b[^\n]*\|\s*iex|Invoke-Expression|\bmv\b[^\n]*\s/|\bMove-Item\b)",
    re.I)
_SYSTEM_DIRS = re.compile(r"^(C:\\Windows|C:\\Program Files|/etc|/usr|/bin|/sbin|/System|/Library)", re.I)


def is_dangerous(name: str, args: dict) -> bool:
    if name in ("run_command",):
        return bool(_DANGEROUS.search(str(args.get("command", ""))))
    if name == "run_python":
        code = str(args.get("code", ""))
        return bool(re.search(r"shutil\.rmtree|os\.remove|os\.unlink|os\.rmdir|subprocess|os\.system|winreg|ctypes", code))
    if name == "write_file":
        return bool(_SYSTEM_DIRS.search(str(args.get("path", ""))))
    if name == "gui_action":
        keys = [str(k).lower() for k in (args.get("keys") or [])]
        return "alt" in keys and "f4" in keys
    return False


def summarize(name: str, args: dict) -> str:
    if name == "run_command":
        return f"{args.get('shell', 'auto')}> {args.get('command', '')}"[:400]
    if name == "run_python":
        return "python:\n" + str(args.get("code", ""))[:400]
    if name == "write_file":
        return f"write {args.get('path')} ({len(str(args.get('content', '')))} chars)"
    if name == "gui_action":
        return f"gui {args.get('action')} {args.get('x', '')},{args.get('y', '')} {args.get('text', '') or args.get('keys', '')}"[:200]
    return f"{name} {args}"[:300]


def _truncate(s: str, limit: int = 16000) -> str:
    if len(s) <= limit:
        return s
    head = s[: limit * 2 // 3]
    tail = s[-(limit // 3):]
    return f"{head}\n… [{len(s) - limit} chars skipped] …\n{tail}"


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        import psutil
        p = psutil.Process(proc.pid)
        for c in p.children(recursive=True):
            try:
                c.kill()
            except Exception:
                pass
        p.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# ------------------------------------------------------------------- shell / python
def default_shell() -> str:
    if IS_WIN:
        return "powershell"
    return "zsh" if IS_MAC and shutil.which("zsh") else ("bash" if shutil.which("bash") else "sh")


def _shell_argv(shell: str, command: str) -> list[str]:
    if shell == "powershell":
        exe = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
        prelude = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; $ProgressPreference='SilentlyContinue'; "
        return [exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", prelude + command]
    if shell == "cmd":
        return ["cmd.exe", "/d", "/s", "/c", "chcp 65001>nul & " + command]
    exe = shutil.which(shell) or shell
    if IS_WIN and shell in ("bash", "sh") and not shutil.which(shell):
        for cand in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
            if Path(cand).exists():
                exe = cand
                break
    return [exe, "-lc", command]


_PS_WRAPPER_RE = re.compile(r"^ *(?:powershell|pwsh)(?:[.]exe)? +(?:-[A-Za-z]+ +)*?-[cC](?:ommand)? +(.*)$", re.S)


def _unwrap_powershell(command: str) -> str:
    """Models like to write `powershell -Command "..."` although run_command already IS PowerShell;
    the nested quoting then breaks. Strip that wrapper and its outer quotes."""
    m = _PS_WRAPPER_RE.match(command)
    if not m:
        return command
    rest = m.group(1).strip()
    if len(rest) >= 2 and rest[0] == rest[-1] and rest[0] in ('"', "'"):
        rest = rest[1:-1]
    return rest or command


def run_command(args: dict) -> dict:
    command = str(args.get("command") or "").strip()
    if not command:
        return {"error": "command is empty"}
    shell = str(args.get("shell") or "auto").lower()
    if shell == "auto":
        shell = default_shell()
    if IS_WIN and shell in ("zsh",):
        shell = "powershell"
    timeout = max(1, min(int(args.get("timeout") or 60), 600))
    cwd = args.get("cwd") or None
    if cwd:
        cwd = os.path.expanduser(str(cwd))
        if not os.path.isdir(cwd):
            return {"error": f"cwd does not exist: {cwd}"}
    if shell == "powershell":
        command = _unwrap_powershell(command)
    argv = _shell_argv(shell, command)
    t0 = time.time()
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, cwd=cwd,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except FileNotFoundError as e:
        return {"error": f"shell not found: {e}"}
    try:
        out, err = proc.communicate(timeout=timeout)
        code = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        out, err = proc.communicate()
        code = -1
        timed_out = True
    dec = lambda b: b.decode("utf-8", "replace") if b else ""  # noqa: E731
    stdout, stderr = dec(out), dec(err)
    if IS_WIN and shell == "cmd" and "\ufffd" in stdout:
        stdout = out.decode("cp866", "replace")
    res = {"shell": shell, "exit_code": code, "stdout": _truncate(stdout.strip()), "stderr": _truncate(stderr.strip(), 6000),
           "elapsed_s": round(time.time() - t0, 2)}
    if timed_out:
        res["error"] = f"timed out after {timeout}s (process killed)"
    return res


def run_python(args: dict) -> dict:
    code = str(args.get("code") or "")
    if not code.strip():
        return {"error": "code is empty"}
    timeout = max(1, min(int(args.get("timeout") or 60), 600))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        path = f.name
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        proc = subprocess.Popen([sys.executable, "-X", "utf8", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                stdin=subprocess.DEVNULL, env=env, cwd=str(Path.home()),
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            out, err = proc.communicate(timeout=timeout)
            code_ = proc.returncode
            res = {"exit_code": code_, "stdout": _truncate(out.decode("utf-8", "replace").strip()),
                   "stderr": _truncate(err.decode("utf-8", "replace").strip(), 6000), "elapsed_s": round(time.time() - t0, 2)}
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            res = {"error": f"timed out after {timeout}s"}
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
    return res


# ------------------------------------------------------------------- files
def read_file(args: dict) -> dict:
    path = Path(os.path.expanduser(str(args.get("path") or ""))).expanduser()
    if not path.is_file():
        return {"error": f"not a file: {path}"}
    limit = max(200, min(int(args.get("max_chars") or 20000), 200000))
    data = path.read_bytes()
    if b"\x00" in data[:4096]:
        return {"path": str(path), "binary": True, "size": len(data)}
    text = data.decode("utf-8", "replace")
    return {"path": str(path), "size": len(data), "content": text[:limit], "truncated": len(text) > limit}


def write_file(args: dict) -> dict:
    path = Path(os.path.expanduser(str(args.get("path") or "")))
    content = str(args.get("content") or "")
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.get("append") else "w"
    with open(path, mode, encoding="utf-8") as f:
        f.write(content)
    return {"path": str(path), "bytes": len(content.encode("utf-8")), "append": bool(args.get("append"))}


def list_directory(args: dict) -> dict:
    path = Path(os.path.expanduser(str(args.get("path") or str(Path.home()))))
    if not path.is_dir():
        return {"error": f"not a directory: {path}"}
    items = []
    for p in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:500]:
        try:
            st = p.stat()
            items.append({"name": p.name, "dir": p.is_dir(), "size": None if p.is_dir() else st.st_size,
                          "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))})
        except Exception:
            items.append({"name": p.name, "dir": p.is_dir()})
    return {"path": str(path), "count": len(items), "items": items}


# ------------------------------------------------------------------- GUI
class _Gui:
    def __init__(self) -> None:
        self._pg = None
        self.shot_scale = 1.0
        self.shot_offset = (0, 0)
        self.shot_maps: dict[int, tuple[float, tuple[int, int]]] = {}   # monitor number -> (scale, (left, top))
        self.last_monitor: Optional[int] = None

    @property
    def pg(self):
        if self._pg is None:
            import pyautogui
            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0.02
            self._pg = pyautogui
        return self._pg

    def screen_size(self) -> tuple[int, int]:
        try:
            w, h = self.pg.size()
            return int(w), int(h)
        except Exception:
            return (0, 0)

    @staticmethod
    def _physical(sct) -> list[dict]:
        """mss lists the whole virtual screen first, then each monitor; we number the monitors from 1."""
        mons = sct.monitors
        return mons[1:] if len(mons) > 1 else mons[:1]

    def list_monitors(self) -> list[dict]:
        """Monitors as the settings panel shows them: number, size, position, which one is primary."""
        try:
            import mss
            with mss.mss() as sct:
                mons = self._physical(sct)
        except Exception:  # noqa: BLE001
            return []
        return [{"index": i, "width": int(m.get("width", 0)), "height": int(m.get("height", 0)),
                 "left": int(m.get("left", 0)), "top": int(m.get("top", 0)),
                 "primary": int(m.get("left", 0)) == 0 and int(m.get("top", 0)) == 0}
                for i, m in enumerate(mons, 1)]

    def screenshot(self, monitor: Optional[int] = None, monitors: Optional[list] = None) -> dict:
        """One PNG per monitor: the explicit `monitor` number if given, otherwise the monitors chosen in the
        settings (SCREENSHOT_MONITORS; empty = all of them)."""
        import mss
        from PIL import Image
        with mss.mss() as sct:
            mons = self._physical(sct)
            n = len(mons)
            try:
                want_one = int(monitor) if monitor is not None else 0
            except (TypeError, ValueError):
                want_one = 0
            if 0 < want_one <= n:
                wanted = [want_one]
            else:
                chosen = monitors if monitors is not None else settings.SCREENSHOT_MONITORS
                wanted = sorted({int(i) for i in (chosen or []) if str(i).isdigit() and 0 < int(i) <= n}) or list(range(1, n + 1))
            shots = []
            for i in wanted:
                mon = mons[i - 1]
                raw = sct.grab(mon)
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                ow, oh = img.size
                scale = 1.0
                max_side = settings.SCREENSHOT_MAX_SIDE
                if max_side and max(ow, oh) > max_side:
                    scale = max(ow, oh) / max_side
                    img = img.resize((round(ow / scale), round(oh / scale)), Image.LANCZOS)
                # gui_action coordinates come back in this frame: remember how to map them onto this monitor
                self.shot_maps[i] = (scale, (int(mon.get("left", 0)), int(mon.get("top", 0))))
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=False, compress_level=3)
                shots.append({"png_base64": base64.b64encode(buf.getvalue()).decode(), "width": img.size[0],
                              "height": img.size[1], "monitor": i, "scale": scale, "screen": [ow, oh]})
        if shots:
            self.last_monitor = shots[0]["monitor"]
            self.shot_scale, self.shot_offset = self.shot_maps[self.last_monitor]
        return {"shots": shots, "monitors": n, "selected": wanted}

    def _map(self, x, y, monitor=None) -> tuple[int, int]:
        """Screenshot-frame pixels -> real screen pixels of the monitor the screenshot was taken from."""
        try:
            key = int(monitor) if monitor is not None else self.last_monitor
        except (TypeError, ValueError):
            key = self.last_monitor
        scale, (ox, oy) = self.shot_maps.get(key, (self.shot_scale, self.shot_offset))
        return int(round(float(x) * scale)) + ox, int(round(float(y) * scale)) + oy

    def action(self, args: dict) -> dict:
        pg = self.pg
        act = str(args.get("action") or "").lower()
        x, y = args.get("x"), args.get("y")
        has_xy = x is not None and y is not None
        if act in ("click", "double_click", "right_click", "move", "drag") and not has_xy:
            return {"error": f"{act} needs x and y"}
        mon = args.get("monitor")   # which monitor's screenshot the coordinates refer to (several monitors)
        if act == "click":
            sx, sy = self._map(x, y, mon)
            pg.click(sx, sy)
        elif act == "double_click":
            sx, sy = self._map(x, y, mon)
            pg.doubleClick(sx, sy)
        elif act == "right_click":
            sx, sy = self._map(x, y, mon)
            pg.rightClick(sx, sy)
        elif act == "move":
            sx, sy = self._map(x, y, mon)
            pg.moveTo(sx, sy, duration=0.1)
        elif act == "drag":
            sx, sy = self._map(x, y, mon)
            tx, ty = self._map(args.get("to_x", x), args.get("to_y", y), mon)
            pg.moveTo(sx, sy)
            pg.dragTo(tx, ty, duration=0.4, button="left")
        elif act == "type":
            text = str(args.get("text") or "")
            if not text:
                return {"error": "text is empty"}
            if text.isascii():
                pg.write(text, interval=0.01)
            else:  # unicode: paste through the clipboard (pyautogui cannot type it)
                import pyperclip
                old = None
                try:
                    old = pyperclip.paste()
                except Exception:
                    pass
                pyperclip.copy(text)
                pg.hotkey("command" if IS_MAC else "ctrl", "v")
                time.sleep(0.15)
                if old is not None:
                    try:
                        pyperclip.copy(old)
                    except Exception:
                        pass
        elif act in ("hotkey", "press"):
            keys = [str(k).lower() for k in (args.get("keys") or [])]
            if not keys:
                return {"error": "keys are empty"}
            if act == "hotkey" and len(keys) > 1:
                pg.hotkey(*keys)
            else:
                for k in keys:
                    pg.press(k)
        elif act == "scroll":
            amount = int(args.get("amount") or -3)
            if has_xy:
                sx, sy = self._map(x, y, mon)
                pg.scroll(amount, x=sx, y=sy)
            else:
                pg.scroll(amount)
        else:
            return {"error": f"unknown action {act}"}
        px, py = pg.position()
        return {"ok": True, "action": act, "mouse": [int(px), int(py)]}


GUI = _Gui()


def open_target(args: dict) -> dict:
    target = str(args.get("target") or "").strip()
    if not target:
        return {"error": "target is empty"}
    if re.match(r"^https?://|^www\.", target, re.I):
        import webbrowser
        webbrowser.open(target if target.lower().startswith("http") else "https://" + target)
        return {"ok": True, "opened": target, "kind": "url"}
    p = Path(os.path.expanduser(target))
    if p.exists():
        if IS_WIN:
            os.startfile(str(p))  # type: ignore[attr-defined]
        elif IS_MAC:
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])
        return {"ok": True, "opened": str(p), "kind": "path"}
    # program name
    try:
        if IS_WIN:
            subprocess.Popen(f'start "" {target}', shell=True)
        elif IS_MAC:
            subprocess.Popen(["open", "-a", target])
        else:
            subprocess.Popen(target, shell=True)
        return {"ok": True, "opened": target, "kind": "program"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"cannot open {target}: {e}"}


def clipboard(args: dict) -> dict:
    import pyperclip
    act = str(args.get("action") or "get")
    if act == "set":
        pyperclip.copy(str(args.get("text") or ""))
        return {"ok": True}
    return {"text": pyperclip.paste()[:20000]}


def list_windows(args: dict) -> dict:
    focus = args.get("focus")
    try:
        import pygetwindow as gw  # type: ignore
    except Exception:
        return {"error": "window listing is only supported on Windows (pygetwindow)"}
    titles = [t for t in gw.getAllTitles() if t.strip()]
    active = None
    try:
        active = gw.getActiveWindow().title
    except Exception:
        pass
    if focus:
        for w in gw.getAllWindows():
            if focus.lower() in (w.title or "").lower():
                try:
                    if w.isMinimized:
                        w.restore()
                    w.activate()
                except Exception as e:  # noqa: BLE001
                    return {"error": f"cannot focus {w.title!r}: {e}"}
                return {"ok": True, "focused": w.title, "box": [w.left, w.top, w.width, w.height]}
        return {"error": f"no window contains {focus!r}", "windows": titles[:60]}
    return {"active": active, "windows": titles[:80]}


def system_info(args: dict) -> dict:
    import psutil
    vm = psutil.virtual_memory()
    disks = []
    for part in psutil.disk_partitions(all=False)[:8]:
        try:
            u = psutil.disk_usage(part.mountpoint)
            disks.append({"mount": part.mountpoint, "total_gb": round(u.total / 2**30, 1), "free_gb": round(u.free / 2**30, 1)})
        except Exception:
            pass
    procs = []
    for p in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            procs.append((p.info["memory_info"].rss if p.info["memory_info"] else 0, p.info["pid"], p.info["name"]))
        except Exception:
            pass
    procs.sort(reverse=True)
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})", "machine": platform.machine(),
        "hostname": platform.node(), "user": getpass.getuser(), "home": str(Path.home()), "cwd": os.getcwd(),
        "python": sys.version.split()[0], "cpu_count": os.cpu_count(), "cpu_percent": psutil.cpu_percent(interval=0.2),
        "ram_total_gb": round(vm.total / 2**30, 1), "ram_used_pct": vm.percent, "disks": disks,
        "screen": list(GUI.screen_size()), "uptime_h": round((time.time() - psutil.boot_time()) / 3600, 1),
        "top_processes_by_memory": [{"pid": pid, "name": n, "rss_mb": round(r / 2**20)} for r, pid, n in procs[:12]],
    }


def client_description() -> dict:
    """Facts about this machine that go into the model's system prompt."""
    w, h = GUI.screen_size()
    tz = time.strftime("%Z")
    offset = "+00:00"
    try:
        import datetime as _dt
        now = _dt.datetime.now().astimezone()
        tz = str(now.tzinfo)
        off = now.utcoffset() or _dt.timedelta(0)
        total = int(off.total_seconds())
        sign = "+" if total >= 0 else "-"
        total = abs(total)
        offset = f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"
    except Exception:
        pass
    return {
        "os": f"{platform.system()} {platform.release()}", "hostname": platform.node(), "user": getpass.getuser(),
        "shell": default_shell(), "screen": f"{w}x{h}" if w else "unknown", "timezone": tz, "utc_offset": offset,
        "python": sys.version.split()[0], "home": str(Path.home()), "tools_enabled": settings.TOOLS_ENABLED,
    }


TOOLS: dict[str, Callable[[dict], dict]] = {
    "run_command": run_command,
    "run_python": run_python,
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "gui_action": GUI.action,
    "open_target": open_target,
    "clipboard": clipboard,
    "list_windows": list_windows,
    "system_info": system_info,
    "__screenshot": lambda a: GUI.screenshot(a.get("monitor"), a.get("monitors")),
}


def execute(name: str, args: dict) -> dict:
    fn = TOOLS.get(name)
    if not fn:
        return {"error": f"tool {name} is not implemented on this client"}
    try:
        return fn(args or {})
    except Exception as e:  # noqa: BLE001
        log.exception("tool %s failed", name)
        return {"error": f"{type(e).__name__}: {e}"}
