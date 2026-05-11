import numpy as np

from kinema.elementary_transforms import ET, ETS, SE3
from kinema.link_connected import LinkConnected

def check_ets_cycle(ets: ETS, q0: np.ndarray, max_err: float = 1e-10):
    ets_loop_delta = np.max(
        np.abs(
            ets.delta_se3(q0, SE3())
        )
    )
    if ets_loop_delta > max_err:
        raise RuntimeError(f"|(A*B*C)^-1 * (A*B*C) - I| = {ets_loop_delta} is too big!")

class TLinkConnected(LinkConnected):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._process_new_cycles()

    def _process_new_cycles(self):
        q0 = self.collect_q0()
        for ets_cycle in self._ets_cycles:
            check_ets_cycle(ets_cycle, q0)

def test_basic_ets_checks():
    ets_base_to_world = ET.Rx(np.pi/4) * ET.tx()
    base = TLinkConnected(ets_base_to_world)
    ets_child_to_base = ET.Rz(np.pi/2) * ET.ty()
    child = TLinkConnected(ets_child_to_base, base)

    ets_child_to_world = ets_base_to_world * ets_child_to_base
    qindex_counter = 0
    for et in ets_child_to_world.joints():
        et.qindex = qindex_counter
        qindex_counter += 1

    q_test = np.array((1,1))
    check_ets_cycle(ets_child_to_world.inv() * child.get_ets_to_base(), q_test)

def test_link_cycles_static():
    def test_impl(flip: bool, use_flexible_cycles: bool = False):
        dT = ET.Rz(-2 * np.pi/3) * ET.tx(1)
        link0 = TLinkConnected(dT)
        link1 = TLinkConnected(dT, link0)
        if flip:
            link2 = TLinkConnected([dT, dT.inv()], [link1, link0], use_flexible_cycles=use_flexible_cycles)
        else:
            link2 = TLinkConnected([dT.inv(), dT], [link0, link1], use_flexible_cycles=use_flexible_cycles)
            
    test_impl(True)
    test_impl(False)
    test_impl(True, True)

def test_link_cycles_parametric():
    dT = ET.Rz() * ET.tx()
    d_q = [-2 * np.pi/3, 1]
    d_q_r = [1, -2 * np.pi/3]
    link0 = TLinkConnected(dT, None, [d_q,])
    link1 = TLinkConnected(dT, link0, [d_q,])
    link2 = TLinkConnected([dT, dT.inv()], [link1, link0], [d_q, d_q_r])

    q0 = np.array(d_q + d_q + d_q + d_q_r )
    assert np.all(link0.collect_q0() == q0 )
    link0.reset_q0(q0)
    assert np.all(link0.collect_q0() == q0 )
