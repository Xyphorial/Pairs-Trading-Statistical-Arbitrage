"""Economic accounting and future-data invariance checks."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from project import signals, backtest, metrics

class EngineTests(unittest.TestCase):
    def setUp(self):
        self.idx = pd.bdate_range('2024-01-01',periods=6)
        self.cfg = dict(commission_bps=2,slippage_bps=3,borrow_bps_year=0)
        self.params = dict(entry=2.,exit=.5,stop=4.,holding=20)

    def test_execution_delay_and_roundtrip_cost(self):
        p = pd.DataFrame({'Y':100.,'X':100.},index=self.idx)
        s = pd.DataFrame({'z':[0.,-3.,-3.,0.,0.,0.],'beta':1.},index=self.idx)
        d,t = backtest(p,s,str(self.idx[0].date()),str(self.idx[-1].date()),self.params,self.cfg)
        self.assertEqual(d.state.tolist(),[0,0,1,1,0,0])
        self.assertAlmostEqual(d.cost.sum(),.001)
        self.assertAlmostEqual(d.nav.iloc[-1],.999)
        self.assertEqual(len(t),1)
        self.assertAlmostEqual(t.net_pnl.sum(),-.001)

    def test_fixed_units_profit_and_trade_reconciliation(self):
        p = pd.DataFrame({'Y':[100,100,100,110,120,120],'X':100.},index=self.idx)
        s = pd.DataFrame({'z':[-3.,-3.,-3.,-3.,-3.,-3.],'beta':1.},index=self.idx)
        d,t = backtest(p,s,'2024-01-01','2024-02-01',self.params,self.cfg)
        self.assertAlmostEqual(d.nav.iloc[-1],1+.1-.0005-.00055)
        self.assertAlmostEqual(t.net_pnl.sum(),d.nav.iloc[-1]-1)
        self.assertEqual(t.iloc[0].reason,'period_end')

    def test_future_prices_cannot_change_past_signals(self):
        rng = np.random.default_rng(2)
        p = pd.DataFrame(np.exp(rng.normal(size=(100,2))*.03+4),index=pd.bdate_range('2024-01-01',periods=100))
        a = signals(p,20)
        p.iloc[70:] *= 3
        b = signals(p,20)
        pd.testing.assert_frame_equal(a.iloc[:70],b.iloc[:70])

    def test_initial_loss_is_in_drawdown(self):
        d = pd.DataFrame({'net_return':[-.1,0,0],'turnover':0.})
        self.assertAlmostEqual(metrics(d)['max_drawdown'],-.1)

if __name__=='__main__': unittest.main()
