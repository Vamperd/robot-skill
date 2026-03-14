import argparse
import json
from pathlib import Path


def build_bar(value, max_value, color):
    width = 0 if max_value <= 0 else max(2, int((value / max_value) * 260))
    return f'<div style="background:{color};height:16px;width:{width}px;border-radius:4px;"></div>'


def make_html(data):
    results = data.get("results", [])
    max_reward = max((item["total_reward"] for item in results), default=1.0)
    rows = []
    for item in results:
        success_text = "成功" if item["success"] else "失败"
        success_color = "#16a34a" if item["success"] else "#dc2626"
        dfa_ratio = item["final_dfa_state"] / 3.0
        rows.append(
            f"""
            <tr>
                <td>{item['scenario']}</td>
                <td style="color:{success_color};font-weight:600;">{success_text}</td>
                <td>{item['steps']}</td>
                <td>{item['total_reward']:.2f}</td>
                <td>{item['final_dfa_state']}/3</td>
                <td>{build_bar(item['total_reward'], max_reward, '#3b82f6')}</td>
                <td>{build_bar(dfa_ratio, 1.0, '#f59e0b')}</td>
            </tr>
            """
        )

    return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <title>Transfer Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; background: #f8fafc; color: #0f172a; }}
    .card {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08); }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 16px; }}
    .metric {{ background: #eef2ff; border-radius: 10px; padding: 16px; }}
    .metric .label {{ color: #475569; font-size: 14px; }}
    .metric .value {{ font-size: 28px; font-weight: 700; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ padding: 12px; border-bottom: 1px solid #e2e8f0; text-align: left; vertical-align: middle; }}
    th {{ background: #f1f5f9; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>迁移评估可视化报告</h1>
    <p>模型：{data.get('model_path', '')}</p>
    <div class="grid">
      <div class="metric"><div class="label">成功率</div><div class="value">{data.get('success_rate', 0.0):.0%}</div></div>
      <div class="metric"><div class="label">平均回报</div><div class="value">{data.get('avg_reward', 0.0):.1f}</div></div>
      <div class="metric"><div class="label">平均最终 DFA</div><div class="value">{data.get('avg_final_dfa_state', 0.0):.2f}</div></div>
      <div class="metric"><div class="label">测试场景数</div><div class="value">{len(results)}</div></div>
    </div>
  </div>

  <div class="card">
    <h2>场景明细</h2>
    <table>
      <thead>
        <tr>
          <th>Scenario</th>
          <th>结果</th>
          <th>步数</th>
          <th>回报</th>
          <th>DFA</th>
          <th>回报条形图</th>
          <th>DFA 进度</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="将迁移评估 JSON 生成 HTML 可视化报告")
    parser.add_argument("--input", default="transfer_eval_results.json", help="评估结果 JSON 路径")
    parser.add_argument("--output", default="transfer_eval_report.html", help="输出 HTML 路径")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    html = make_html(data)
    Path(args.output).write_text(html, encoding="utf-8")
    print(f"已生成可视化报告: {args.output}")


if __name__ == "__main__":
    main()