#!/usr/bin/env python3
"""
embed-pro live dashboard — like htop for your embedding server.
Usage: python dashboard.py [http://localhost:8020] [--refresh 1.0]
"""
import sys, time, argparse, json, urllib.request, urllib.error
from collections import deque
from datetime import datetime, timedelta

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.bar import Bar
from rich import box

# --- Config ---

HISTORY_LEN = 60  # data points for sparklines
SPARK_CHARS = "▁▂▃▄▅▆▇█"


def spark(values: list[float]) -> str:
    if not values or max(values) == 0:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    return "".join(SPARK_CHARS[min(int((v - mn) / rng * 7), 7)] for v in values)


def fmt_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    if seconds < 86400:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h}h {m}m"
    d = seconds // 86400
    h = (seconds % 86400) // 3600
    return f"{d}d {h}h"


def fmt_mem(mb: int) -> str:
    if mb < 1024:
        return f"{mb} MB"
    return f"{mb / 1024:.1f} GB"


def fetch(url: str, path: str) -> dict:
    try:
        r = urllib.request.urlopen(f"{url}{path}", timeout=3)
        return json.loads(r.read())
    except Exception:
        return {}


class Dashboard:
    def __init__(self, url: str):
        self.url = url
        self.console = Console()

        # History buffers
        self.embed_rps = deque(maxlen=HISTORY_LEN)
        self.rerank_rps = deque(maxlen=HISTORY_LEN)
        self.mem_history = deque(maxlen=HISTORY_LEN)
        self.inflight_history = deque(maxlen=HISTORY_LEN)
        self.cache_history = deque(maxlen=HISTORY_LEN)

        self.prev_embed = 0
        self.prev_rerank = 0
        self.prev_time = time.time()
        self.errors = 0
        self.last_health = {}
        self.last_metrics_text = ""

    def poll(self):
        health = fetch(self.url, "/health")
        if not health:
            self.errors += 1
            return

        self.errors = 0
        self.last_health = health
        now = time.time()
        dt = now - self.prev_time

        # Calculate RPS
        embed_total = health.get("embed_requests", 0)
        rerank_total = health.get("rerank_requests", 0)
        if dt > 0:
            self.embed_rps.append((embed_total - self.prev_embed) / dt)
            self.rerank_rps.append((rerank_total - self.prev_rerank) / dt)
        self.prev_embed = embed_total
        self.prev_rerank = rerank_total
        self.prev_time = now

        self.mem_history.append(health.get("memory_mb", 0))
        self.inflight_history.append(health.get("inflight_requests", 0))
        self.cache_history.append(health.get("cache_size", 0))

        # Fetch raw metrics for extra detail
        try:
            r = urllib.request.urlopen(f"{self.url}/metrics", timeout=3)
            self.last_metrics_text = r.read().decode()
        except Exception:
            pass

    def _parse_metric(self, name: str, labels: str = "") -> float:
        for line in self.last_metrics_text.split("\n"):
            if line.startswith(name):
                if labels and labels not in line:
                    continue
                try:
                    return float(line.split()[-1])
                except (ValueError, IndexError):
                    pass
        return 0.0

    def _parse_histogram_avg(self, name: str) -> float:
        total = self._parse_metric(f"{name}_sum")
        count = self._parse_metric(f"{name}_count")
        if count > 0:
            return total / count
        return 0.0

    def build(self) -> Layout:
        h = self.last_health
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )
        layout["left"].split_column(
            Layout(name="models", size=7),
            Layout(name="traffic"),
        )
        layout["right"].split_column(
            Layout(name="resources", size=11),
            Layout(name="graphs"),
        )

        # Header
        status = h.get("status", "unknown")
        if not h:
            status_style = "bold red"
            status_text = "OFFLINE"
        elif status == "ok":
            status_style = "bold green"
            status_text = "RUNNING"
        elif status == "draining":
            status_style = "bold yellow"
            status_text = "DRAINING"
        else:
            status_style = "bold red"
            status_text = status.upper()

        uptime = fmt_duration(h.get("uptime_seconds", 0))
        backend = h.get("backend", "?")
        device = h.get("device", "?")

        header = Table.grid(padding=1)
        header.add_column(ratio=1)
        header.add_column(justify="center", ratio=2)
        header.add_column(justify="right", ratio=1)
        header.add_row(
            Text(f" embed-pro", style="bold"),
            Text(f"{status_text}", style=status_style),
            Text(f"{backend}/{device}  up {uptime} ", style="dim"),
        )
        layout["header"].update(Panel(header, box=box.HEAVY))

        # Models
        models_table = Table(box=None, show_header=False, padding=(0, 2))
        models_table.add_column("Model", style="bold")
        models_table.add_column("Status")
        models_table.add_column("Info", style="dim")

        embed_loaded = h.get("embed_loaded", False)
        rerank_loaded = h.get("rerank_loaded", False)
        models_table.add_row(
            "bge-m3",
            Text("loaded", style="green") if embed_loaded else Text("idle", style="dim"),
            "embedding  1024d  8192 tok",
        )
        models_table.add_row(
            "bge-reranker-v2-m3",
            Text("loaded", style="green") if rerank_loaded else Text("idle", style="dim"),
            "reranker  8192 tok",
        )
        layout["models"].update(Panel(models_table, title="[bold cyan]Models", border_style="cyan"))

        # Traffic
        embed_total = h.get("embed_requests", 0)
        rerank_total = h.get("rerank_requests", 0)
        inflight = h.get("inflight_requests", 0)
        ws = h.get("ws_connections", 0)

        embed_avg_ms = self._parse_histogram_avg("embed_latency_seconds") * 1000
        rerank_avg_ms = self._parse_histogram_avg("rerank_latency_seconds") * 1000

        err_503 = self._parse_metric("embed_errors_total", 'status="503"') + self._parse_metric("rerank_errors_total", 'status="503"')
        err_500 = self._parse_metric("embed_errors_total", 'status="500"') + self._parse_metric("rerank_errors_total", 'status="500"')
        err_504 = self._parse_metric("embed_errors_total", 'status="504"') + self._parse_metric("rerank_errors_total", 'status="504"')
        rejected = self._parse_metric("rejected_requests_total", 'reason="backpressure"')
        deduped = self._parse_metric("embed_dedup_saved_total")

        curr_embed_rps = self.embed_rps[-1] if self.embed_rps else 0
        curr_rerank_rps = self.rerank_rps[-1] if self.rerank_rps else 0

        traffic_table = Table(box=None, show_header=True, padding=(0, 2))
        traffic_table.add_column("", style="bold")
        traffic_table.add_column("Total", justify="right")
        traffic_table.add_column("RPS", justify="right", style="cyan")
        traffic_table.add_column("Avg", justify="right", style="yellow")
        traffic_table.add_column("Spark", min_width=20)

        traffic_table.add_row(
            "Embed",
            str(embed_total),
            f"{curr_embed_rps:.1f}",
            f"{embed_avg_ms:.0f}ms" if embed_avg_ms > 0 else "-",
            Text(spark(list(self.embed_rps)), style="green"),
        )
        traffic_table.add_row(
            "Rerank",
            str(rerank_total),
            f"{curr_rerank_rps:.1f}",
            f"{rerank_avg_ms:.0f}ms" if rerank_avg_ms > 0 else "-",
            Text(spark(list(self.rerank_rps)), style="blue"),
        )
        traffic_table.add_row("", "", "", "", "")
        traffic_table.add_row("In-flight", str(inflight), "", "", Text(spark(list(self.inflight_history)), style="yellow"))
        traffic_table.add_row("WebSocket", str(ws), "", "", "")

        if err_500 + err_503 + err_504 + rejected > 0:
            traffic_table.add_row("", "", "", "", "")
            if err_500 > 0:
                traffic_table.add_row(Text("Errors 500", style="red"), Text(f"{int(err_500)}", style="red"), "", "", "")
            if err_503 > 0:
                traffic_table.add_row(Text("Errors 503", style="red"), Text(f"{int(err_503)}", style="red"), "", "", "")
            if err_504 > 0:
                traffic_table.add_row(Text("Timeouts", style="red"), Text(f"{int(err_504)}", style="red"), "", "", "")
            if rejected > 0:
                traffic_table.add_row(Text("Rejected 429", style="yellow"), Text(f"{int(rejected)}", style="yellow"), "", "", "")

        if deduped > 0:
            traffic_table.add_row("Deduped", f"{int(deduped)}", "", "", "")

        layout["traffic"].update(Panel(traffic_table, title="[bold cyan]Traffic", border_style="cyan"))

        # Resources
        mem_mb = h.get("memory_mb", 0)
        cache_size = h.get("cache_size", 0)
        cache_cap = h.get("cache_capacity", 1024)
        cache_pct = (cache_size / cache_cap * 100) if cache_cap > 0 else 0

        res_table = Table(box=None, show_header=False, padding=(0, 2))
        res_table.add_column("", style="bold", width=12)
        res_table.add_column("", width=12)
        res_table.add_column("", ratio=1)

        mem_style = "green" if mem_mb < 2048 else ("yellow" if mem_mb < 4096 else "red")
        res_table.add_row("Memory", Text(fmt_mem(mem_mb), style=mem_style), Text(spark(list(self.mem_history)), style=mem_style))
        res_table.add_row(
            "Cache",
            f"{cache_size}/{cache_cap}",
            Text(spark(list(self.cache_history)), style="cyan"),
        )
        cache_bar_text = f"  {'█' * int(cache_pct / 5)}{'░' * (20 - int(cache_pct / 5))} {cache_pct:.0f}%"
        res_table.add_row("", "", Text(cache_bar_text, style="cyan"))

        cache_hits = self._parse_metric("embed_cache_hits_total")
        cache_misses = self._parse_metric("embed_cache_misses_total")
        hit_rate = cache_hits / (cache_hits + cache_misses) * 100 if (cache_hits + cache_misses) > 0 else 0
        res_table.add_row("Hit rate", f"{hit_rate:.1f}%", "")

        otel = "on" if h.get("otel") else "off"
        res_table.add_row("OTel", otel, "")

        layout["resources"].update(Panel(res_table, title="[bold cyan]Resources", border_style="cyan"))

        # Graphs panel — RPS history
        graphs_table = Table(box=None, show_header=False, padding=(0, 1))
        graphs_table.add_column("", width=8, style="bold")
        graphs_table.add_column("", ratio=1)

        embed_rps_list = list(self.embed_rps)
        rerank_rps_list = list(self.rerank_rps)
        mem_list = list(self.mem_history)

        # Bigger sparkline blocks
        def wide_spark(values: list[float], width: int = 50) -> str:
            if not values:
                return ""
            # Resample to target width
            if len(values) >= width:
                step = len(values) / width
                resampled = [values[int(i * step)] for i in range(width)]
            else:
                resampled = values
            return spark(resampled)

        max_embed_rps = max(embed_rps_list) if embed_rps_list else 0
        max_rerank_rps = max(rerank_rps_list) if rerank_rps_list else 0
        max_mem = max(mem_list) if mem_list else 0

        graphs_table.add_row(
            Text("Embed", style="green"),
            Text(wide_spark(embed_rps_list, 50), style="green") + Text(f"  peak {max_embed_rps:.1f}/s", style="dim"),
        )
        graphs_table.add_row("", "")
        graphs_table.add_row(
            Text("Rerank", style="blue"),
            Text(wide_spark(rerank_rps_list, 50), style="blue") + Text(f"  peak {max_rerank_rps:.1f}/s", style="dim"),
        )
        graphs_table.add_row("", "")
        graphs_table.add_row(
            Text("Memory", style="yellow"),
            Text(wide_spark(mem_list, 50), style="yellow") + Text(f"  peak {fmt_mem(int(max_mem))}", style="dim"),
        )

        layout["graphs"].update(Panel(graphs_table, title="[bold cyan]History (60s)", border_style="cyan"))

        # Footer
        now_str = datetime.now().strftime("%H:%M:%S")
        footer = Table.grid(padding=1)
        footer.add_column(ratio=1)
        footer.add_column(justify="right")
        footer.add_row(
            Text(f" {self.url}", style="dim"),
            Text(f"q=quit  {now_str} ", style="dim"),
        )
        layout["footer"].update(Panel(footer, box=box.HEAVY))

        return layout


def main():
    parser = argparse.ArgumentParser(description="embed-pro live dashboard")
    parser.add_argument("url", nargs="?", default="http://localhost:8020", help="Server URL")
    parser.add_argument("--refresh", "-r", type=float, default=1.0, help="Refresh interval (seconds)")
    args = parser.parse_args()

    dash = Dashboard(args.url)
    console = Console()

    # Initial poll
    dash.poll()

    try:
        with Live(dash.build(), console=console, refresh_per_second=2, screen=True) as live:
            while True:
                time.sleep(args.refresh)
                dash.poll()
                live.update(dash.build())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
