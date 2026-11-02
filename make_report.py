"""Build a self-contained Chinese HTML report from saved, frozen results."""
import base64
import html
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

def build(folder):
    if not (folder/'run_metadata.json').exists():
        status = (folder/'STATUS.txt').read_text()
        tests = pd.read_csv(folder/'cointegration_tests.csv')
        text = '<!doctype html><meta charset="utf-8"><title>配对交易研究结果</title><style>body{max-width:1000px;margin:50px auto;font:17px/1.7 system-ui}td,th{padding:8px;border-bottom:1px solid #ddd}</style><h1>配对交易研究：未选出合格策略</h1>'
        text += '<p>固定的协整、FDR 与验证期约束下，没有合格策略。不能据此声称策略盈利，也没有为了取得结果放宽筛选条件。</p><pre>'+html.escape(status)+'</pre><h2>形成期最小 p 值的股票对</h2>'
        text += tests.head(20).to_html(index=False,border=0)
        text += '<p>所有检验见 cointegration_tests.csv，参数表见 validation_sensitivity.csv。完整方法、数据覆盖和限制见 README_中文.md 及 data/download_audit.csv。没有最终交易策略，因此不提供虚构的交易日志与收益曲线。</p>'
        path = ROOT/'Pairs_Trading_Report.html'
        path.write_text(text)
        print(path)
        return
    meta = json.loads((folder/'run_metadata.json').read_text())
    perf = pd.read_csv(folder/'performance.csv')
    costs = pd.read_csv(folder/'test_cost_stress.csv')
    selected = meta['selected_pairs']
    tests = pd.read_csv(folder/'cointegration_tests.csv')
    pairs = pd.concat([tests[(tests.y==y)&(tests.x==x)] for y,x in selected])
    def image(name):
        p=folder/name
        return '<img alt="'+html.escape(name)+'" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
    def table(df): return df.to_html(index=False,border=0,float_format=lambda v:f'{v:.4f}')
    rows=[]
    for _,r in perf.iterrows():
        rows.append({'区间':r['period'],'累计收益':f'{r.total_return:.2%}','年化收益':f'{r.annualized_return:.2%}',
                     '年化波动':f'{r.annualized_volatility:.2%}','Sharpe':f'{r.sharpe:.2f}',
                     '最大回撤':f'{r.max_drawdown:.2%}','交易数':int(r.trades),'胜率':f'{r.win_rate:.1%}'})
    tr=pd.read_csv(folder/'test_trades.csv')
    concentration = tr.groupby('pair').agg(net_pnl=('net_pnl','sum'),trades=('net_pnl','size'))
    concentration['net_pnl'] /= len(selected)
    test=perf[perf.period=='test'].iloc[0]
    oos = pd.read_csv(folder/'test_cointegration_diagnostic.csv')
    kind = meta['data_kind']
    title = '模拟数据：仅软件检查' if kind.startswith('SYNTHETIC') else '配对交易 · 样本外研究'
    conclusion = ('本次最终测试期为正收益，但不能据此排除幸存者偏差与选择偏差。' if test.total_return>0 else '本次最终测试期亏损，现有设置不支持策略具有稳定盈利能力。')
    conclusion += f' 测试期只有 {int(test.trades)} 笔交易，且全部收益来自 {len(selected)} 对股票，统计证据有限。'
    if (oos.test_coint_p > .05).any():
        conclusion += ' 至少一个入选股票对在最终测试期未通过 5% 水平的协整检验，不能认为协整关系已得到样本外确认。'
    audit = ROOT/'data/download_audit.csv'
    coverage = ''
    if audit.exists():
        a=pd.read_csv(audit); coverage=f'请求 {len(a)} 个代码，成功返回 {int(a.received.sum())} 个；formation 完整历史筛选后 {meta["formation_stocks"]} 个。下载失败代码见 data/download_audit.csv。'
    doc = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>{title}</title>
<style>body{{max-width:1100px;margin:60px auto;padding:0 24px;font:17px/1.8 system-ui;color:#18313b;background:#f5f5f0}}h1{{font-size:44px;letter-spacing:-1px}}h2{{margin-top:42px}}.tag{{color:#216e8c;font-weight:bold;letter-spacing:2px}}.callout{{padding:22px 28px;background:#dfebe8;border-left:5px solid #216e8c}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}th,td{{padding:10px;text-align:right;border-bottom:1px solid #ddd}}th{{background:#e4eceb}}img{{width:100%;background:white;margin:18px 0;border-radius:10px}}pre{{padding:20px;background:#e4eceb;white-space:pre-wrap}}small{{color:#53636a}}</style>
<div class="tag">STATISTICAL ARBITRAGE / RESEARCH NOTE</div><h1>{title}</h1>
<p>形成期 2015–2019 · 参数验证 2020–2022 · 最终测试 2023–2025</p>
<div class="callout">{conclusion} 本报告展示费用后结果；没有为了测试期收益重新调参。</div>
<h2>01 / 数据与实验设计</h2><p>{coverage} 数据为 Yahoo 复权收盘价。当前成分股回溯历史存在幸存者偏差。本研究不是无偏的历史时点 S&P 500 指数策略。</p>
<p>形成期完成 {meta['formation_tests']:,} 次协整检验。固定回归方向，使用 Engle–Granger p 值与 BH q 值。先建立互不重叠的候选股票对，再用验证期净 Sharpe 选参数。最终测试不参与选择。</p>
<h2>02 / 选中的股票对</h2>{table(pairs[['y','x','p_value','q_value','beta','correlation']])}
{image('candidate_pvalues.png')}
<h2>03 / 冻结的参数与交易规则</h2><pre>{html.escape(json.dumps(meta['selected_parameters'],indent=2))}</pre>
<p>信号收盘后生成，下一个收盘成交。美元敞口比例 1 : beta，入场总敞口为 sleeve 资金的 100%；持仓后保持调整单位。等初始资金组合，不每日再平衡。佣金 2 bp + 滑点 3 bp / 单边实际成交额，空头融券费年化 1%。现金利率设为零。</p>
<h2>04 / 样本内与样本外</h2>{table(pd.DataFrame(rows))}{image('equity_curve.png')}
<p>形成期和验证期结果包含筛选/调参影响，不能作为独立泛化证据。表中每个区间独立从净值 1、空仓开始，期末强制平仓并计费用。</p>
<h2>05 / 成本与收益集中度</h2>{table(costs[['execution_cost_multiplier','annualized_return','sharpe','max_drawdown']])}
<p>执行成本倍数仅改变佣金与滑点，融券费保持不变。以下是各股票对测试期净利润对初始等权组合资金的贡献；可判断收益是否集中。</p>{table(concentration.reset_index())}
<h2>06 / 样本外协整与交易信号</h2>{table(oos)}<p>这里的 p 值只用于事后诊断。p 大于 0.05 表示在 5% 水平不能拒绝不协整的原假设；不是证明两者一定不协整，也不允许用此结果回头挑选更好看的股票对。</p>'''
    for y,x in selected: doc += image(f'spread_z_{y}_{x}.png')
    doc += '''<h2>07 / 如何复现与审计</h2><p>阅读 README_中文.md；在文件夹中运行 python project.py --prices data/adjusted_close.csv。逐笔交易见 test_trades.csv，参数全网格见 validation_sensitivity.csv，样本外协整诊断见 test_cointegration_diagnostic.csv。run_metadata.json 保存配置和输入 SHA-256，requirements-lock.txt 保存核心依赖版本。</p>
<p>限制：缺少退市股票与历史成分股变更、动态借券/召回、真实买卖价差与冲击、保证金与融资限制。调整收盘价与调整单位是总回报近似，并非真实成交账本。协整关系可能失效；统计显著不等于经济可交易。</p>
<small>资料：用户提供的项目 brief；statsmodels.coint 官方文档；yfinance.download 官方文档。数据来源与方法链接详见 README。</small></html>'''
    path=ROOT/'Pairs_Trading_Report.html'
    path.write_text(doc)
    print(path)

if __name__=='__main__':
    import sys
    build(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/real_data')
