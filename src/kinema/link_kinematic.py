# Robot Kinematics Playground (Python Library)
# Author: anton.gudym@gmail.com (Anton Gudym)

from typing import Optional, Union, List, Tuple

import time

import numpy as np
import scipy

from kinema.elementary_transforms import ET, ETS, SE3

from kinema.link_connected import LinkConnected

from kinema.optimization import MinimizerLevenbergMarquardt, Termination

class Manipulability():
    """ 
    General ability to move or rotate in an arbitrary direction
    """
    def __init__(self, J: np.ndarray):
        """
        J is 6 by N jacobian matrix, where N is dimensionality of configuration space
        """
        self._JJT = J @ J.T # 6x6 is more compact, then 6xN in general
    
    def calc_yoshikawa(self, only2d:bool=False) -> tuple[float, float]:
        """
        Calculate translational and rotational manipulability(by Yoshikawa)

        Returns
        -------
        m_x = sqrt(det(J_x @ J_x.T)), where _x states for rotational or translational part of Jacobian
        """
        if only2d:
            m_t = np.sqrt(np.abs(np.linalg.det(self._JJT[:2,:2])))
            m_r = np.sqrt(np.abs(self._JJT[5,5]))
        else:
            m_t = np.sqrt(np.abs(np.linalg.det(self._JJT[:3,:3])))
            m_r = np.sqrt(np.abs(np.linalg.det(self._JJT[3:,3:])))
        return m_t, m_r

    def calc_singvalues(self) -> np.ndarray:
        _, s, _ = np.linalg.svd(self._JJT)
        return s

class IKResult:
    """ Description of inverse-kinematics process """
    def __init__(self,
                q: np.ndarray,
                q0: np.ndarray,
                constraints_error: np.ndarray,
                desc: str,
                convergence_time_sec: float):
        self.q = q
        self.q0 = q0
        self.constraints_error = constraints_error
        self.desc = desc
        self.convergence_time_sec = convergence_time_sec

    @property
    def e(self):
        """ Translational and rotational errors per X constraint of shape [2, X] """
        return np.vstack((
            np.linalg.norm(self.constraints_error[:3, :],axis=0),
            np.linalg.norm(self.constraints_error[3:, :],axis=0)))
    
    def __str__(self):
        e = self.e
        return  f"{self.desc}Total time {self.convergence_time_sec:.3f} sec."
                #f"\nError (T, R): {e[0]}, {e[1]}" \
                #f"\n||q0-q_sol||l2 {np.linalg.norm(self.q0 - self.q):.3f}" \
                #f"\nSolution {self.q}"

class IKConstraint:
    def __init__( self,
                ets: ETS,
                num_joints: int,
                T_goal: Optional[SE3] = None,
                weights: Optional[list[float]] = None):
        self._ets = ets

        self.num_joints = num_joints
        self._J = np.zeros((LinkConnected.num_task_dims, num_joints))

        if T_goal is None:
            self._T_goal_mat = SE3()
        else:
            self._T_goal_mat = T_goal

        if weights is None:
            self._W = np.eye(LinkConnected.num_task_dims)
        else:
            if len(weights) != LinkConnected.num_task_dims:
                raise ValueError(f"Invalid weights number {len(weights)} != {LinkConnected.num_task_dims}")
            self._W = np.diag(weights)

    def update_target_pose(self, T_goal: SE3):
        """
        Update SE3 transformation before the solve
        """
        if not isinstance(T_goal, SE3):
            raise ValueError("Invalid type!")

        self._T_goal_mat = T_goal

    def calc_residuals(self, q: np.ndarray) :
        return self._W @ self._T_goal_mat.calc_delta(self._ets.eval(q))
    
    def calc_jacobian(self, q: np.ndarray) :
        self._J = self._ets.jacob0(q)
        return self._W @ self._J
    
class LinkKinematic(LinkConnected):
    """
    Connected link extended with kinematic equations
    """
    # Tolerance of the kinematic deviations, i.e. maximal deviation considered as 0
    tolerance:float = 1e-6

    rigid_body_dofs:int = 6

    def __init__(self, *args, debug: bool=False, **kwargs):
        """
        Parameters
        ----------
        debug
            Extended logging on/off flag
        *args
            `LinkConnected` properties
        **kwargs
            `LinkConnected` properties
        """
        self._debug = debug

        super().__init__(*args, **kwargs)
        
        # Add by the user with public interface, can be erased
        self._constraints_user: List[IKConstraint] = []
        # Added automatically when kinematical cycle is detected, permanent
        self._constraints_auto: List[IKConstraint] = []

    @property
    def constraints(self):
        return self._constraints_auto + self._constraints_user

    def correct_initial_configuration(self):
        """
        Verify and fix kinematics when all the links are constructed.

        Search for kinematic cycles in the transformation sequence, verify that initial
        configuration is consistent with it. If not - try fix, throws ValueError in 
        case of errors.
        """
        q0 = self.collect_q0()
        enorms = []
        for link in self._list_all_links():
            for ets_cycle in link._ets_cycles:
                link._constraints_auto.append(IKConstraint(ets_cycle, link.num_joints))
                enorms.append(np.linalg.norm(link._constraints_auto[-1].calc_residuals(q0)))

        # Apply cycle constraint:
        # Kinematic cycles are zero transformation for correct configuration
        if any(e > LinkKinematic.tolerance for e in enorms):
            ik_result = self.solve(q0)
            if self._debug:
                print(f"Kinematic chain is not accurate, error max-norm is: {np.max(enorms)}."
                      f" Adjusting default configuration:\n{ik_result}" )
            self.reset_q0(ik_result.q)
        else:
            if self._debug:
                if len(enorms) > 0:
                    print(f"All {len(enorms)} circular kinematic chains are accurate.")
                else:
                    print("Kinematic cycles not found.")

    def add_constraint( self,
                  T_goal: SE3,
                  weights: Optional[list[float]] = None,
                  link_base: Union["LinkConnected", None] = None,
                  reset: bool = False) -> IKConstraint :
        """
        Prepare data for kinematic constraints to optimize in `LinkKinematic.solve`.

        Parameters
        ----------
        T_goal 
            desired 6-DoF orientation in either world or some specific link's coordinate frame
        weights
            six-dimensional vector applied to delta between actual and goal orientations.
            Default value is [1,1,1,1,1,1].
        link_base
            defines coordinate frame in which T_goal is defined, default value corresponds to the World frame
        reset
            clear previous constraints, except auto-added, e.g. closed kinematic connection

        Returns
        -------
        A newly created `IKConstraint`
        """
        if not isinstance(T_goal, SE3):
            raise ValueError("Invalid type!")
        
        if reset:
            self._constraints_user.clear()

        self._constraints_user.append(
            IKConstraint(self.get_ets_to_base(link_base), self.num_joints, T_goal, weights))
        return self._constraints_user[-1]

    def solve(self, q0: Optional[np.ndarray] = None, iterations_max: int=25):
        """
        Find new configuration for all connected links, each link can have specific constraints.

        Parameters
        ----------
        q0 
            initial estimation, if `None` then default `self.collect_q0()` is used
        iterations_max
            max amount of non-linear optimization iterations
        """
        total_time = time.time()
        if q0 is None:
            q0 = self.collect_q0()

        def calc_residuals(q:np.ndarray) :
            residuals = []
            for link in self._list_all_links():
                for constraint in link.constraints:
                    residuals.append(constraint.calc_residuals(q))
            return np.concatenate(residuals)
        
        def calc_jacobian(q:np.ndarray) :
            jacobians = []
            for link in self._list_all_links():
                for constraint in link.constraints:
                    jacobians.append(constraint.calc_jacobian(q))
            J = np.concatenate(jacobians, axis=0)
            return scipy.sparse.lil_array(J)

        minimizer = MinimizerLevenbergMarquardt(iterations_max=iterations_max)
        q, (minimizer_desc, termination, minimizer_state) = minimizer.run(q0, calc_residuals, calc_jacobian)
        
        constraints_error = calc_residuals(q).reshape((-1, self.rigid_body_dofs)).T
        
        if termination == Termination.IterationsLimit:
            raise ValueError(f"Kinematics optimization didn't converge!\n{minimizer_desc}")

        return IKResult(q, q0, constraints_error, minimizer_desc, time.time() - total_time)

def generate_multicycles_robot( LinkClass,
                               geometry: Tuple=(),
                               link_length: float=1,
                               num_cycles: int = 3,
                               use_flexible_cycles: bool = True,
                               debug: bool = False ):
    links = []
    link0 = LinkClass(*geometry)
    for section_id in range(num_cycles):
        link1 = LinkClass( *geometry,
                            ets_l2ps=ET.Rz(),
                            link_parents=link0,
                            q0_l2ps=np.pi/4 if section_id & 1 else 3 * np.pi/4)
        link2 = LinkClass( *geometry,
                            ets_l2ps=ET.tx(link_length) * ET.Rz(),
                            link_parents=link0,
                            q0_l2ps=np.pi/4 if section_id & 1 else 3 * np.pi/4)
        link3 = LinkClass( *geometry,
                            ets_l2ps=[ET.tx(link_length) * ET.Rz(), ET.tx(link_length) * ET.Rz() * ET.tx(-link_length)],
                            link_parents=[link1, link2],
                            q0_l2ps=[-np.pi/2,-np.pi/2],
                            use_flexible_cycles=use_flexible_cycles,
                            debug=debug)
        links += [link0, link1, link2, link3]
        link0 = link3
    links[-1].correct_initial_configuration()
    return links
