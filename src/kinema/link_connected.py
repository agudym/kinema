# Robot Kinematics Playground (Python Library)
# Author: anton.gudym@gmail.com (Anton Gudym)

from typing import Optional, Union, List, Dict

import numpy as np

from roboticstoolbox import ET, ETS

import networkx as nx

class LinkConnected:
    """
    Robotic link with elementary transformations sequence (ETS) SE(3) chain

    ETS defines relations(parametrized orientation) between links coordinate frames,
    e.g. having 2 connected links link_A(parent) and link_B(child), if link_A's 
    coordinate frame is same as World coordinate frame, then it's 
    ETS: ets_link_A_to_World = Identity_SE(3) or 0-transformation, while if link_B rotates
    around link_B then ETS: ets_link_B_to_link_A = ETS.Rz() (defines 1 angular parameter).
    Position of link_B's points in the world:
    pt_world = ets_link_A_to_World @ ets_link_B_to_link_A(angle) @ pt_link_B
    Note that link's geometry is not defined here, only basic coordinate frames
    """
    # Dimensionality of the task space which is se(3) algebra
    num_task_dims = 6

    def __init__(self,
                 ets_l2ps: Optional[Union[List[ETS], ETS, ET]] = None,
                 link_parents: Optional[Union[List["LinkConnected"], "LinkConnected"]] = None,
                 q0_l2ps: Optional[Union[List[List[float]], List[float], float]] = None,
                 name: Optional[str] = None,
                 use_flexible_cycles: bool = True):
        """
        Parameters
        ----------
            ets_l2ps
                transformation against each parent or the world coordinate frame
            link_parents
                list of related(connected) links
            q0_l2ps
                default configuration of the link, will automatically correct parameters
                for closed kinematic chains
            name
                string description of the link, defaults to "link{id}" format
            use_flexible_cycles
                detected cycles are not constrained for all joints, i.e. all parameters
                can be optimized for example, otherwise only new joints are parametrized.
                Turned on might decrease performance, however more adaptive.
        """
        self._name = name
        self._use_flexible_cycles = use_flexible_cycles

        if ets_l2ps is None:
            self._ets_l2ps: List[ETS] = [ETS(), ]
        elif isinstance(ets_l2ps, list):
            self._ets_l2ps: List[ETS] = []
            for ets in ets_l2ps:
                 self._ets_l2ps.append(ETS(ets))
        else:
            self._ets_l2ps: List[ETS] = [ETS(ets_l2ps), ]

        if link_parents is None:
            self._link_parents: List["LinkConnected"] = []
        elif isinstance(link_parents, list):
            self._link_parents: List["LinkConnected"] = link_parents
        else:
            self._link_parents: List["LinkConnected"] = [link_parents, ]

        self._q0_l2ps: List[np.ndarray] = []
        if q0_l2ps is None:
            for ets_l2p in self._ets_l2ps:
                self._q0_l2ps.append(np.zeros(ets_l2p.n, dtype=float))
        else:    
            if isinstance(q0_l2ps, float) or isinstance(q0_l2ps, int):
                q0_l2ps = [[q0_l2ps,],]
            
            if not isinstance(q0_l2ps, list) :
                raise ValueError("Initial configuration is not a List.")
            
            if len(q0_l2ps) != len(self._ets_l2ps):
                raise ValueError("Initial configuration has wrong length."
                                 "Should be a List per each parent link.")

            for q0_l2p, ets_l2p in zip(q0_l2ps, self._ets_l2ps):
                if isinstance(q0_l2p, list):
                    self._q0_l2ps.append(np.array(q0_l2p, dtype=float))
                else:
                    self._q0_l2ps.append(np.array([q0_l2p,], dtype=float))
                if ets_l2p.n != self._q0_l2ps[-1].size:
                    raise ValueError("Initial configuration's parameters does not match ETS.")            

        self._is_base_link: bool = len(self._link_parents) == 0        
        if self._is_base_link:
            if len(self._ets_l2ps) != 1:
                raise ValueError("The base link's ets_l2ps should have size 1.")
        else:
            if len(self._link_parents) != len(self._ets_l2ps) or len(self._q0_l2ps) > 0 and len(self._q0_l2ps) != len(self._link_parents):
                raise ValueError("Input lists have different size.")

        self._generate_unique_ids()
        self._process_connection_graph()

    @property
    def num_links(self):
        return self._root._num_links
    
    @property
    def num_joints(self):
        return self._root._num_joints
    
    @property
    def name(self) -> str:
        """ Unique name of the link in the sequence """
        return f"link{self._link_id}" if self._name is None else self._name

    @staticmethod
    def get_joint_indices(ets: ETS) -> List[int]:
        """
        Indices(positions) of the according ET(from ETS) in global configuration vector
        """
        joint_indices: List[int] = []
        for et in ets:
            if not et.isjoint:
                continue
            joint_indices.append(et.jindex)
        return joint_indices
    
    def get_orientation(self, q: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Calculate link to base 6DoF orientation, using initial configuration.
        Parameters
        ----------
        q
            configuration of robot's links (q.size equals to Robot's dofs); if None
            initial configuration is used
        Returns
        -------
            4x4 Euclidean (SE3) transformation matrix
        """
        if q is None:
            q = self.collect_q0()
        return self.get_ets_to_base().eval(q)

    def get_ets_to_base(self, link_base: Union["LinkConnected", None] = None) -> ETS:
        """
        Get ETS describing complete, shortest kinematics chain between current link and the base

        Parameters
        ----------
        link_base
            Means world coordinate frame if None, or specific link otherwise

        Returns
        -------
            ETS - transformation between current link and link_base
        """
        ets_base_to_world = ETS()
        
        if link_base is None:
            link_base = self._root
            # kinematics in the world frame
            ets_base_to_world = link_base._ets_l2ps[0]
        
        # note the order: from base to the actual node(self), due to
        # T_child_to_base = ... * T_child * T_child_of_child * ...
        ets_l2b = self._get_ets_between(link_base, self)

        return ets_base_to_world * ets_l2b
    
    def collect_q0(self, check_init:bool = True) -> np.ndarray:
        """
        Returns initial configuration of ALL links
        Parameters
        ----------
        check_init
            verify output is free of NaNs
        """
        q0 = np.empty(self.num_joints)# full parameter list
        q0.fill(np.nan)
        for link in self._list_all_links():
            for q0_l2p, ets_l2p in zip(link._q0_l2ps, link._ets_l2ps):
                j = 0
                for et in ets_l2p:
                    if not et.isjoint:
                        continue
                    q0[et.jindex] = q0_l2p[j]
                    j += 1
        if check_init and np.sum(np.isnan(q0)) > 0:
            raise RuntimeError("Uninitialized configuration parameters detected.")
        return q0

    def reset_q0(self, q0_new: np.ndarray) -> None:
        """
        Reset initial configuration
        """
        if self.num_joints != q0_new.size:
            raise ValueError("Invalid size of the new configuration vector.")
        if np.sum(np.isnan(q0_new)) > 0:
            raise RuntimeError("Uninitialized configuration parameters detected.")
        for link in self._list_all_links():
            for i, ets_l2p in enumerate(link._ets_l2ps):
                j = 0
                for et in ets_l2p:
                    if not et.isjoint:
                        continue
                    link._q0_l2ps[i][j] = q0_new[et.jindex]
                    j += 1

    def _list_all_links(self):
        return self._graph.nodes()

    def _get_ets_between(self, link_src: "LinkConnected", link_dst: "LinkConnected") -> ETS:
        path: Union[List["LinkConnected"], Dict] = nx.shortest_path(self._graph, link_src, link_dst)
        ets = ETS()
        for i in range(len(path) - 1):
            graph_edge = self._graph.edges[path[i], path[i+1]]
            ets = ets * graph_edge['ets']
        return ets.compile()

    def _generate_unique_ids(self) -> None:
        if self._is_base_link:
            self._num_links = 0
            # Joint is a 1-dof connection
            self._num_joints = 0
            self._root: "LinkConnected" = self
        else:
            self._root: "LinkConnected" = self._link_parents[0]._root

        self._link_id: int = self._root._num_links
        self._root._num_links += 1

        # assign unique kinematic parameter (or joint) ID
        # interesting feature https://github.com/petercorke/robotics-toolbox-python/issues/393
        for ets_l2p in self._ets_l2ps:
            for et in ets_l2p:
                if not et.isjoint:
                    continue
                et.jindex = self._root._num_joints
                self._root._num_joints += 1

    def _process_connection_graph(self):
        if self._is_base_link:
            self._graph: nx.DiGraph = nx.DiGraph()
        else:
            self._graph: nx.DiGraph = self._root._graph # common reference

        # Goes before the graph update to avoid cycles search (which are about to appear)
        self._find_new_cycles()

        if self._is_base_link:
            self._graph.add_node(self)
        else:
            for link_parent, ets_l2p, q0_l2p in zip(self._link_parents, self._ets_l2ps, self._q0_l2ps):
                # ets is the transformation from second node to the first
                self._graph.add_edge(link_parent, self, ets=ets_l2p, q0=q0_l2p)
                q0_reversed = np.flip(q0_l2p)
                self._graph.add_edge(self, link_parent, ets=ets_l2p.inv(), q0=q0_reversed)

    def _find_new_cycles(self):
        self._ets_cycles: List[ETS] = []
        num_parents = len(self._link_parents)
        if num_parents <= 1:
            return
            
        if not self._use_flexible_cycles:
            q0 = self.collect_q0(check_init=False) # not all q0 are set, because graph is not updated yet
        
        for i in range(num_parents):
            for j in range(i + 1, num_parents):
                ets_path = self._get_ets_between(self._link_parents[i], self._link_parents[j])
                if not self._use_flexible_cycles:
                    ets_path = ETS(ET.SE3(ets_path.eval(q0)))
                ets_cycle = (ets_path * self._ets_l2ps[j] * self._ets_l2ps[i].inv()).compile()
                self._ets_cycles.append(ets_cycle)

    def __str__(self):
        return self.name

def main():
    def check_ets_cycle(ets: ETS, q0: np.ndarray, max_err: float = 1e-10):
        ets_loop_delta = np.max(np.abs(ets.eval(q0) - np.identity(4)))
        if ets_loop_delta > max_err:
            raise RuntimeError(f"|(A*B*C)^-1 * (A*B*C) - I| = {ets_loop_delta} is too big!")
        else:
            print(f"ETS loop error is OK: {ets_loop_delta}")

    class TLinkConnected(LinkConnected):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._process_new_cycles()

        def _process_new_cycles(self):
            q0 = self.collect_q0()
            for ets_cycle in self._ets_cycles:
                check_ets_cycle(ets_cycle, q0)

    # Basic ETS checks
    ets_base_to_world = ET.Rx(np.pi/4) * ET.tx()
    base = TLinkConnected(ets_base_to_world)
    ets_child_to_base = ET.Rz(np.pi/2) * ET.ty()
    child = TLinkConnected(ets_child_to_base, base)
    print(f"Elementary transform chain between World and base-link: {base.get_ets_to_base()}")
    print(f"Elementary transform chain between World and end-link: {child.get_ets_to_base()}")
    print(f"Elementary transform chain between base and end links: {child.get_ets_to_base(base)}")

    ets_child_to_world = ets_base_to_world * ets_child_to_base
    jindex_counter = 0
    for et in ets_child_to_world:
        if et.isjoint:
            et.jindex = jindex_counter
            jindex_counter += 1

    q_test = np.array((1,1))#some random values
    check_ets_cycle(ets_child_to_world.inv() * child.get_ets_to_base(), q_test)

    # Test Cycles
    # Non-parametric
    def test_link_cycles_static(flip: bool, use_flexible_cycles: bool = False):
        dT = ET.Rz(-2 * np.pi/3) * ET.tx(1)
        link0 = TLinkConnected(dT)
        link1 = TLinkConnected(dT, link0)
        if flip:
            link2 = TLinkConnected([dT, dT.inv()], [link1, link0], use_flexible_cycles=use_flexible_cycles)
        else:
            link2 = TLinkConnected([dT.inv(), dT], [link0, link1], use_flexible_cycles=use_flexible_cycles)
    test_link_cycles_static(True)
    test_link_cycles_static(False)
    test_link_cycles_static(True, True)

    # Parametric
    dT = ET.Rz() * ET.tx()
    d_q = [-2 * np.pi/3, 1]
    d_q_r = [1, -2 * np.pi/3]
    link0 = TLinkConnected(dT, None, [d_q,])
    link1 = TLinkConnected(dT, link0, [d_q,])
    link2 = TLinkConnected([dT, dT.inv()], [link1, link0], [d_q, d_q_r]) # note reversed order of parameters (due to inv())

    q0 = np.array(d_q + d_q+ d_q + d_q_r )
    if not np.all(link0.collect_q0() == q0 ):
        raise RuntimeError(f"Configuration vectors {q0} and {link0.q0} don't match!")
    link0.reset_q0(q0)
    if not np.all(link0.collect_q0() == q0 ):
        raise RuntimeError(f"Configuration vectors {q0} and {link0.q0} don't match!")

    print("Test passed!")

if __name__ == "__main__":
    main()
