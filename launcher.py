#!/usr/bin/env python3
"""
LLM Monitor - Launcher
======================
A single-port landing page that mounts the Court Dashboard and
Multi-Agent Dashboard as FastAPI sub-applications.

All three UIs are served from one host:port:
    /            - Landing page
    /court/      - Court Dashboard
    /dashboard/  - Multi-Agent Dashboard
"""

import os
import argparse
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="LLM Monitor")

COURT_PATH = "/court"
DASHBOARD_PATH = "/dashboard"


def get_landing_html():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Behavioral Monitor</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0a0a0f;
    color: #e0e0e0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}}

.header {{
    text-align: center;
    margin-bottom: 60px;
}}
.header h1 {{
    font-size: 2.4rem;
    font-weight: 300;
    letter-spacing: 2px;
    color: #f0f0f0;
    margin-bottom: 12px;
}}
.header p {{
    color: #888;
    font-size: 1rem;
    letter-spacing: 1px;
}}

.cards {{
    display: flex;
    gap: 40px;
    flex-wrap: wrap;
    justify-content: center;
    max-width: 900px;
}}

.card {{
    background: #12121a;
    border: 1px solid #2a2a3a;
    border-radius: 16px;
    width: 380px;
    padding: 40px 32px;
    text-decoration: none;
    color: inherit;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}}
.card:hover {{
    border-color: #555;
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}}
.card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}}
.card-court::before {{ background: linear-gradient(90deg, #c9a44a, #e8c860); }}
.card-dashboard::before {{ background: linear-gradient(90deg, #4a9ac9, #60b8e8); }}

.card-icon {{
    font-size: 3rem;
    margin-bottom: 20px;
    display: block;
}}
.card h2 {{
    font-size: 1.3rem;
    font-weight: 500;
    margin-bottom: 12px;
}}
.card-court h2 {{ color: #e8c860; }}
.card-dashboard h2 {{ color: #60b8e8; }}

.card p {{
    color: #888;
    font-size: 0.85rem;
    line-height: 1.6;
    margin-bottom: 16px;
}}

.card-features {{
    list-style: none;
    padding: 0;
    margin-bottom: 24px;
}}
.card-features li {{
    color: #999;
    font-size: 0.78rem;
    padding: 4px 0;
    padding-left: 16px;
    position: relative;
}}
.card-features li::before {{
    content: '>';
    position: absolute;
    left: 0;
    color: #555;
}}

.card-link {{
    display: inline-block;
    font-size: 0.8rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 8px 0;
}}
.card-court .card-link {{ color: #e8c860; }}
.card-dashboard .card-link {{ color: #60b8e8; }}

.footer {{
    margin-top: 60px;
    color: #444;
    font-size: 0.75rem;
    text-align: center;
}}
.footer a {{ color: #666; text-decoration: none; }}
.footer a:hover {{ color: #999; }}

.status {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}}
.status-live {{ background: #4caf50; box-shadow: 0 0 6px #4caf50; }}
.status-off {{ background: #555; }}
</style>
</head>
<body>

<div class="header">
    <h1>LLM Behavioral Monitor</h1>
    <p>Probe-based activation steering and multi-agent analysis</p>
</div>

<div class="cards">
    <a href="{COURT_PATH}/" class="card card-court" id="courtCard">
        <span class="card-icon">&#x2696;&#xfe0f;</span>
        <h2>Federal Court Simulation</h2>
        <p>Fixed-role courtroom simulation with judge, prosecution, defense, and jury agents in a structured legal proceeding.</p>
        <ul class="card-features">
            <li>Pre-configured court agents with legal personas</li>
            <li>Phase-based proceedings (opening, evidence, deliberation)</li>
            <li>Gated injection with dynamic scaling</li>
            <li>Per-token behavioral heatmaps</li>
            <li>DNA fingerprinting and SAE analysis</li>
        </ul>
        <span class="card-link"><span class="status status-off" id="courtStatus"></span>Open Court Dashboard &rarr;</span>
    </a>

    <a href="{DASHBOARD_PATH}/" class="card card-dashboard" id="dashCard">
        <span class="card-icon">&#x1f9e0;</span>
        <h2>Multi-Agent Monitor</h2>
        <p>Configurable multi-agent interactions with user-selectable roles, topologies, and probe injection targets.</p>
        <ul class="card-features">
            <li>8 agent templates (proposer, critic, judge, etc.)</li>
            <li>6 interaction topologies (debate, panel, adversarial)</li>
            <li>Configurable probe injection per agent</li>
            <li>Global token score heatmaps</li>
            <li>Cross-model galaxy comparison</li>
        </ul>
        <span class="card-link"><span class="status status-off" id="dashStatus"></span>Open Agent Dashboard &rarr;</span>
    </a>
</div>


<script>
async function checkStatus(url, statusId) {{
    try {{
        const res = await fetch(url, {{ signal: AbortSignal.timeout(2000) }});
        document.getElementById(statusId).className = 'status status-live';
    }} catch(e) {{
        document.getElementById(statusId).className = 'status status-off';
    }}
}}
checkStatus('{COURT_PATH}/api/status', 'courtStatus');
checkStatus('{DASHBOARD_PATH}/api/status', 'dashStatus');
setInterval(() => {{
    checkStatus('{COURT_PATH}/api/status', 'courtStatus');
    checkStatus('{DASHBOARD_PATH}/api/status', 'dashStatus');
}}, 5000);
</script>

</body>
</html>"""


# Track which sub-apps are mounted so we can trigger their startup
_mounted_apps = []


@app.on_event("startup")
async def startup():
    """Propagate startup to mounted sub-applications."""
    for sub_app in _mounted_apps:
        # Trigger each sub-app's on_event("startup") handlers
        for handler in sub_app.router.on_startup:
            await handler()


@app.get("/", response_class=HTMLResponse)
async def landing():
    return get_landing_html()


@app.get("/api/status")
async def status():
    return {"status": "ok", "service": "launcher"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Monitor Launcher")
    parser.add_argument("--model", default="Qwen_Qwen2.5-0.5B-Instruct",
                        help="Model name for both dashboards")
    parser.add_argument("--model-dir", default="/drive1/xiacong/models",
                        help="Directory containing local models")
    parser.add_argument("--device", default="cuda:0",
                        help="Device for model inference")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for the unified server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--no-court", action="store_true",
                        help="Don't mount court dashboard")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Don't mount multi-agent dashboard")

    args = parser.parse_args()

    # Set environment variables for sub-app startup hooks
    os.environ["COURT_MODEL"] = args.model
    os.environ["COURT_DEVICE"] = args.device
    os.environ["COURT_MODEL_DIR"] = args.model_dir
    os.environ["DASHBOARD_MODEL"] = args.model
    os.environ["DASHBOARD_DEVICE"] = args.device
    os.environ["DASHBOARD_MODEL_DIR"] = args.model_dir

    print(f"\n{'='*60}")
    print(f"  LLM Behavioral Monitor - Unified Server")
    print(f"{'='*60}")
    print(f"  Model:  {args.model}")
    print(f"  Device: {args.device}")
    print(f"\n  Landing:   http://localhost:{args.port}/")

    if not args.no_court:
        import court_dashboard
        court_dashboard.BASE_PATH = COURT_PATH
        app.mount(COURT_PATH, court_dashboard.app)
        _mounted_apps.append(court_dashboard.app)
        print(f"  Court:     http://localhost:{args.port}{COURT_PATH}/")

    if not args.no_dashboard:
        import dashboard_server
        dashboard_server.BASE_PATH = DASHBOARD_PATH
        app.mount(DASHBOARD_PATH, dashboard_server.app)
        _mounted_apps.append(dashboard_server.app)
        print(f"  Dashboard: http://localhost:{args.port}{DASHBOARD_PATH}/")

    print(f"{'='*60}\n")

    uvicorn.run(app, host=args.host, port=args.port)
