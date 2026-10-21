"""Portable reports built from recorded episodes, with no external assets."""

import csv
from html import escape
import json
from pathlib import Path
from urllib.parse import quote

from .reports import summarize
from .storage import read_json, write_json


def load_run(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    path = root / "episodes.jsonl"
    records = []
    if path.exists():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed episode record at line {number}") from exc
    summary = summarize(records, manifest["expected_episodes"])
    summary["mock_only"] = bool(manifest.get("mock_only") or summary["mock_only"])
    summary["run_status"] = manifest["status"]
    summary["complete"] = summary["complete"] and manifest["status"] == "complete"
    return manifest, records, summary


def number(value, percent=False):
    if value is None:
        return "—"
    return f"{value:.1%}" if percent else f"{value:.2f}"


def csv_text(value):
    text = str(value)
    # CSV can be opened in spreadsheet software; do not interpret task text as a formula.
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text


def video_links(root, paths):
    links = []
    for item in paths:
        relative = Path(item)
        candidate = (root / relative).resolve()
        if relative.is_absolute() or not candidate.is_relative_to(root) or not candidate.is_file():
            continue
        url = quote(candidate.relative_to(root).as_posix(), safe="/")
        links.append(f'<a href="{url}">{escape(relative.name)}</a>')
    return "<br>".join(links) or "—"


def export_report(run_dir):
    root = Path(run_dir).resolve()
    manifest, records, summary = load_run(root)
    write_json(root / "summary.json", summary)
    fields = ["task_id", "initial_state", "seed", "instruction", "first_attempt_success",
              "success", "retries", "steps", "elapsed_s", "reason"]
    with (root / "episodes.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in fields}
            row["steps"] = sum(a["steps"] for a in record["attempts"])
            row["reason"] = record["attempts"][-1]["reason"]
            row["instruction"] = csv_text(row["instruction"])
            row["reason"] = csv_text(row["reason"])
            writer.writerow(row)
    cards = [
        ("已记录场景", f'{summary["episodes"]} / {summary["expected_episodes"]}'),
        ("首次成功率", number(summary["first_attempt_success_rate"], True)),
        ("最终成功率", number(summary["success_rate"], True)),
        ("推理 P95 / ms", number((summary["inference_ms"] or {}).get("p95"))),
    ]
    card_html = "".join(f'<div class="card"><span>{key}</span><strong>{value}</strong></div>'
                        for key, value in cards)
    task_rows = ""
    for task, values in summary["per_task"].items():
        rate = values["successes"] / values["episodes"]
        task_rows += (f'<tr><td>{escape(task)}</td><td>{values["successes"]} / {values["episodes"]}</td>'
                      f'<td><meter min="0" max="1" value="{rate}"></meter> {rate:.1%}</td></tr>')
    episode_rows = ""
    for record in records:
        reason = record["attempts"][-1]["reason"]
        cells = [record.get("task_id"), record.get("initial_state"), record.get("instruction", ""),
                 "成功" if record["success"] else "失败", record["retries"], reason]
        episode_rows += "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in cells)
        episode_rows += f'<td>{video_links(root, record.get("videos", []))}</td></tr>'
    tag = "MOCK · 流程联调" if summary["mock_only"] else "LIBERO · 自定义子集评估"
    state = "完整运行" if summary["complete"] else "部分运行"
    ci = summary["success_rate_ci95"]
    interval = f"{ci[0]:.1%} – {ci[1]:.1%}" if ci else "—"
    metadata = escape(json.dumps({"config": manifest.get("config"),
                                  "provenance": manifest.get("provenance"),
                                  "error": manifest.get("error")}, ensure_ascii=False, indent=2))
    html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MealVLA · Evaluation report</title>
<style>
:root{{color-scheme:light; font-family:system-ui,sans-serif;color:#193047;background:#f3f6fa}}
body{{max-width:1120px;margin:0 auto;padding:36px 24px}}h1{{margin:10px 0;font-size:36px}}
.tag{{color:#166a78;font-weight:600}}.muted{{color:#53677b}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}}
.card,section{{background:white;border:1px solid #dbe3ed;border-radius:12px;padding:20px;margin:20px 0}}
.card span,.card strong{{display:block}}.card strong{{font-size:28px;margin-top:10px}}
section{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{text-align:left;border-bottom:1px solid #e4ebf2;padding:12px 10px;vertical-align:top}}
meter{{width:120px}}a{{color:#076a89}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}
@media(max-width:700px){{.cards{{grid-template-columns:repeat(2,1fr)}}body{{padding:20px 12px}}}}
</style></head><body>
<div class="tag">{tag}</div><h1>MealVLA 评估报告</h1>
<p class="muted">{state} · {escape(manifest['status'])} · {escape(root.name)}</p>
<p><a href="summary.json">指标 JSON</a> · <a href="episodes.csv">逐回合 CSV</a> · <a href="manifest.json">运行配置</a></p>
<div class="cards">{card_html}</div>
<p>最终成功率 Wilson 95% 区间：{interval}。重置重试总数：{summary['reset_retries']}。</p>
<p class="muted">成功率分母为已记录的场景数；重试不增加分母。Mock 结果仅用于流程联调。部分运行不视为完整评估。</p>
<section><h2>任务结果</h2><table><thead><tr><th>任务 ID</th><th>成功 / 场景</th><th>成功率</th></tr></thead><tbody>{task_rows}</tbody></table></section>
<section><h2>逐回合记录</h2><table><thead><tr><th>任务</th><th>初始状态</th><th>指令</th><th>结果</th><th>重试</th><th>原因</th><th>视频</th></tr></thead><tbody>{episode_rows}</tbody></table></section>
<section><details><summary>配置与来源</summary><pre>{metadata}</pre></details></section>
<p class="muted">{escape(summary['protocol'])}</p></body></html>'''
    (root / "report.html").write_text(html, encoding="utf-8")
    return summary
