"""Synthetic unit-test fixtures only; never included in experiment results."""
import unittest

import numpy as np
import pandas as pd

from audit_partial_stage_rankings import select_window, fit_window


def trace(start=94):
    t = np.arange(0, 81, .5)
    soc = start-np.clip(np.floor((t-10)/3),0,None)
    frame=pd.DataFrame(dict(elapsed_time=t,battery=soc,phase='wind_tunnel_hover',mid=5,mission_pad=5,x=0,y=0,h=80))
    frame['is_drop']=frame.battery.diff().lt(0)
    return frame


class WindowTests(unittest.TestCase):
    def test_partial_high_does_not_require_full_charge(self):
        g=trace(94);w=select_window(g,100,82,5,'first',0)
        self.assertEqual(w.battery.iloc[0],93)
        self.assertEqual(w.battery.iloc[-1],88)
        self.assertAlmostEqual(fit_window(w)['raw_rate'],20)

    def test_second_window_is_nonoverlapping(self):
        g=trace(94)
        a=select_window(g,100,82,5,'first',0)
        b=select_window(g,100,82,5,'second',0)
        self.assertGreaterEqual(b.elapsed_time.iloc[0],a.elapsed_time.iloc[-1])
        self.assertLess(b.battery.iloc[-1],a.battery.iloc[-1])

    def test_no_cross_stage_or_fault_bridge(self):
        g=trace(90)
        self.assertIsNone(select_window(g,100,85,5,'first',0))
        g=trace(94);g.loc[g.elapsed_time.between(22,24),'phase']='recoverable_error'
        w=select_window(g,100,82,5,'first',0)
        self.assertGreater(w.elapsed_time.iloc[0],24)

    def test_initial_and_normal_plateaus(self):
        w=select_window(trace(94),100,82,5,'first',0)
        self.assertEqual(w.elapsed_time.iloc[0],13)
        self.assertTrue(w.battery.diff().eq(0).any())
        self.assertTrue(w.is_drop.iloc[-1])

    def test_after_ready(self):
        w=select_window(trace(94),100,82,5,'after_all_ready',25)
        self.assertGreaterEqual(w.elapsed_time.iloc[0],25)


if __name__=='__main__':unittest.main()
