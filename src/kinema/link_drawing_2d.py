# Robot Kinematics Playground (Python Library)
# Author: anton.gudym@gmail.com (Anton Gudym)

from typing import Optional, Union, List, Dict, Tuple
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from roboticstoolbox import ET, ETS

from kinema.link_kinematic import LinkKinematic, generate_multicycles_robot

class LinkDrawing2D(LinkKinematic):
    """
    Auxiliary link sequence's renderer in 2D. Works with 2 types of transformation ET.tx, ET.Rz()
    """
    def __init__(self,
                pt_start:np.ndarray,
                pt_end:np.ndarray,
                link_width: float,
                *args,
                draw_all: bool = True,
                **kwargs):
        """ 
            Creates link with kinematics and geometrical(shape) properties

            Parameters
            ----------
            pt_start
                link's beginning 2D point in it's kinematic coordinate frame (defined by ETS in LinkKinematic)
            pt_end
                link's ending 2D point, the length is the distance between pt_start and pt_end
            link_width
                link's width
            draw_all
                visualize auxiliary info along with the link
            *args : `LinkKinematic` properties
            **kwargs : `LinkKinematic` properties
        """
        if pt_start.size != 2 or pt_end.size != 2:
            raise ValueError("Invalid input point size")

        # Might be required in destructor
        self.fig, self.ax, self._event_callback_cid = None, None, None

        super().__init__(*args, **kwargs)

        # NOTE Build with only LinkDrawing2D types, i.e. "dynamic_cast" below is safe
        for link in self._link_parents:
            if not isinstance(link, type(self)):
                raise ValueError(f"Invalid type {type(link)} is used for initialization.")

        # for ets_l2p in self._ets_l2ps:
        #     if ets_l2p.n > 1:
        #         raise ValueError("Class doesn't support links with >1 joints per connection. "
        #                          "Consider splitting the link.")
        self._prepare_joints_ets()

        # start-end boundaries relative to the parents coordinate frame
        self._pts_bound = np.vstack((np.hstack((pt_start, (0,1))), np.hstack((pt_end, (0,1))))).T
        self.link_width = link_width
        self.draw_all = draw_all
        
        self._ets_l2w = self.get_ets_to_base()
        
        # links colors
        self.joint_color = "r"
        self.link_color = "g"
        self.alpha = 0.5
        
        # list of drawn features
        self._mpl_artists = []


    @property
    def geometry(self):
        return (self._pts_bound[:2,0], self._pts_bound[:2,1], self.link_width)

    @property
    def link_length(self):
        return np.linalg.norm(self._pts_bound[:, 0] - self._pts_bound[:, 1])

    @property
    def joint_radius(self):
        return self.link_width / 2.

    def __del__(self):
        if self.fig is not None:
            self._reset_event_callback()
            plt.close(self.fig)

    def render( self,
                q: Optional[np.ndarray] = None,
                event_callback: Optional[Tuple[str, object]] = None,
                text_str: str = "",
                xlim: Tuple[float, float] = (-10.,10.),
                ylim: Tuple[float, float] = (-10.,10.)) -> None :
        if self.fig is None:
            self._create_fig(xlim, ylim)

        if event_callback is not None: # Don't reset previously set connection
            self._reset_event_callback(event_callback) # "dynamic_cast"

        for link in self._list_all_links():
            if link._link_id > self._link_id: # rendering only "preceding" links
                continue                      # to render everything the last link must be used
            link._clear() # "dynamic_cast"
            link._render_link2d(q, self.ax) # "dynamic_cast"

        if len(text_str) > 0 and self.ax is not None:
            # Move text to bottom right, using Axes coordinates (0,0 is bottom-left, 1,1 is top-right)
            # transform=self.ax.transAxes makes the position independent of xlim/ylim
            content = f"{text_str}\n\nHint: Left/Right-button Drag for Smooth/Random initialization"
            self._mpl_artists.append(self.ax.text(0.98, 0.02, content, 
                                                  ha="right", va="bottom", 
                                                  transform=self.ax.transAxes,
                                                  fontsize=12,
                                                  alpha=self.alpha,
                                                  bbox=dict(facecolor='white', alpha=0.3, edgecolor='none')))

        plt.draw()

    def _render_link2d( self, q: Optional[np.ndarray], ax: Axes) -> None:
        if q is None:
            q = self.collect_q0()
        T_l2w = self._ets_l2w.eval(q)
        pts2d_w = T_l2w @ self._pts_bound

        # Render link
        self._draw_mpl_line(pts2d_w.copy(), self.link_width, self.link_color, ax, self.draw_all)

        # Render joint(s)
        for ets_j2w, et_type in self._ets_j2w_with_type:
            T_j2w = ets_j2w.eval(q)
            joint_origin = T_j2w[:2,3]
            if et_type == 3:
                self._draw_mpl_circle(joint_origin, ax)
            else:
                joint_axis = T_j2w[:2,et_type]
                joint_end = joint_origin + joint_axis * self.link_length
                self._draw_mpl_line(np.vstack((joint_origin, joint_end)).T, self.link_width/2, self.joint_color, ax)

        # Render coordinate frames
        if self.draw_all:
            self._draw_mpl_coordinate_frame(T_l2w, ax)
        
    def _reset_event_callback(self, event_callback: Optional[Tuple[str, object]] = None) -> None:
        if self.fig is None:
            raise RuntimeError("Link doesn't own a figure")
        
        if self._event_callback_cid is not None:
            self.fig.canvas.mpl_disconnect(self._event_callback_cid)
            self._event_callback_cid = None

        if event_callback is not None:
            self._event_callback_cid = self.fig.canvas.mpl_connect(event_callback[0], event_callback[1])

    def _draw_mpl_circle(self, center: np.ndarray, ax: Axes) -> None:
        self._mpl_artists.append(
            ax.add_artist(Circle(tuple(center), self.joint_radius, color=self.joint_color, alpha=self.alpha)))
        
    def _draw_mpl_coordinate_frame(self, transform_l2w_mat: np.ndarray, ax: Axes) -> None:
        l = 1 # arrow length
        pt_start = transform_l2w_mat[:2,3]
        dir_x = transform_l2w_mat[:2,0] * l
        dir_y = transform_l2w_mat[:2,1] * l
        self._mpl_artists += [
            ax.arrow(*pt_start, *dir_x, head_width=0.1, color="r"),
            ax.arrow(*pt_start, *dir_y, head_width=0.1, color="g"),]
    
    def _draw_mpl_line(self, pts: np.ndarray, linewidth:float, linecolor: str, ax: Axes, add_text: bool = False) -> None:
        class LineDataUnits(Line2D):
            """ https://stackoverflow.com/questions/19394505/expand-the-line-with-specified-width-in-data-unit/42972469#42972469 """
            def __init__(self, *args, **kwargs):
                _lw_data = kwargs.pop("linewidth", 1) 
                super().__init__(*args, **kwargs)
                self._lw_data = _lw_data
            def _get_lw(self):
                if self.axes is not None:
                    ppd = 72./self.axes.figure.dpi
                    trans = self.axes.transData.transform
                    return ((trans((1, self._lw_data))-trans((0, 0)))*ppd)[1]
                else:
                    return 1
            def _set_lw(self, lw):
                self._lw_data = lw
            _linewidth = property(_get_lw, _set_lw)
            
        # Link's line width compensation
        d = pts[:,1] - pts[:,0]
        d_n = d / np.linalg.norm(d)
        d = d_n * linewidth / 2
        pts[:,0] += d
        pts[:,1] -= d
        self._mpl_artists.append(
            ax.add_line(LineDataUnits(pts[0], pts[1], c=linecolor, linewidth=linewidth, alpha=self.alpha)))
        if add_text:
            text_loc = np.average(pts[:2,:2], axis=1).flatten() + np.array((-d_n[1], d_n[0])) * linewidth
            self._mpl_artists.append(
                ax.text( text_loc[0], text_loc[1],  f"{self.name}", ha="center", va="center", 
                         rotation=np.rad2deg(np.arctan2(d_n[1], d_n[0])), alpha=self.alpha))

    def _prepare_joints_ets(self):
        # Type correspond to the column of SE3 matrix required for visualization
        self._ets_j2w_with_type: List[Tuple[ETS, int]] = []
        for i, ets_l2p in enumerate(self._ets_l2ps):
            if self._is_base_link:
                ets_p2w = ET.tx(0)
            else:
                ets_p2w = self._link_parents[i].get_ets_to_base()
            ets_j2p = []
            for et in ets_l2p:
                if et.isjoint:
                    if et.isrotation:
                        et_type = 3
                    elif et.axis == "tx":
                        et_type = 0
                    elif et.axis == "ty":
                        et_type = 1
                    elif et.axis == "tz":
                        et_type = 2
                    else:
                        raise RuntimeError("Unknown axis")
                    ets_j2w = (ets_p2w * ETS(ets_j2p)).compile()
                    self._ets_j2w_with_type.append((ets_j2w, et_type))
                ets_j2p.append(et)

    def _clear(self) -> None:
        for artist in self._mpl_artists:
            artist.remove()
        self._mpl_artists.clear()

    def _create_fig(self, xlim: Tuple[float, float], ylim: Tuple[float, float]):
        plt.rcParams.update({'font.size': 20})
        self.fig, self.ax = plt.subplots(1,1)
        self.fig.set_size_inches(19,10)
        self.fig.set_dpi(50)
        self.ax.grid(color="black")
        self.ax.set_aspect(1)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        self.fig.suptitle("Inverse Kinematics Solver: Drag to Set New goal")

    @staticmethod
    def generate_sequential_robot(
        num_links:int,
        link_length: float,
        ets: ETS,
        q0: Optional[List[float]] = None,
        link_base: Optional["LinkDrawing2D"] =  None,
        draw_all: Union[List[bool], bool] = True,
        link_width: float = 0.5) -> List["LinkDrawing2D"]:
        """
        Make a sequence of connected links

        Parameters 
        ----------
        num_links
            total amount of output links
        link_length
            geometrical length of the link, it's also a distance between joints
        ets
            description of the transformation between a pair of links
        q0
            optional initial configuration for the ets
        link_base
            optional base link, `None` for World coordinate frame
        draw_all
            render all kind of auxiliary info, like link title, coordinate frame etc.
        link_width
            geometrical width of the link
        
        Returns
        -------
            Connected links sequence 
        """
        pt_start = np.array((-link_length,0))
        # it's more convenient to have end of the link coincide with the origin
        pt_end = np.array((0,0))
        args = (pt_start, pt_end, link_width)

        links = [link_base, ]
        for i in range(num_links):
            if isinstance(draw_all, list):
                draw_all_i = draw_all[i]
            else:
                draw_all_i = draw_all
            links.append( LinkDrawing2D(*args, ets, links[-1], q0, draw_all=draw_all_i) )
        return links[1:]

    @staticmethod
    def make_hexagon_robot(
            joint_type: str = "R",
            closed_kinematics:bool = True,
            use_flexible_cycles:bool = False,
            **kwargs) -> List["LinkDrawing2D"]:
        """
        Make a closed kinematic chain

        Parameters
        ----------
        joint_type
            "R" for rotational, "P" for prismatic joints
        closed_kinematics
            switch between sequential(open) and closed(+1 link) kinematics
        use_flexible_cycles
            optimize(if necessary) whole configuration, see `LinkConnected.__init__`
        **kwargs
            `LinkDrawing2D.generate_sequential_robot` arguments
        """
        def check_draw_all():
            if "draw_all" in kwargs:
                if isinstance(kwargs["draw_all"], list):
                    return kwargs["draw_all"][-1]
                else:
                    return kwargs["draw_all"]
            else:
                return True
        angle_z = -np.pi/3
        link_length = 5
        if joint_type == "R":
            dT =  ET.Rz() * ET.tx(link_length)
            links = LinkDrawing2D.generate_sequential_robot(5, 5, dT, angle_z, **kwargs)
            if closed_kinematics:
                links.append(LinkDrawing2D(*(links[-1].geometry), [dT,dT.inv()], [links[-1], links[0]],
                debug=True, use_flexible_cycles=use_flexible_cycles, draw_all=check_draw_all()))
        elif joint_type == "P":
            dT = ET.Rz(angle_z) * ET.tx() * ET.tx(link_length)
            links = LinkDrawing2D.generate_sequential_robot(5, 5, dT, link_length/2, **kwargs)
            if closed_kinematics:
                links.append(LinkDrawing2D(*(links[-1].geometry), [dT, dT.inv()], [links[-1], links[0]],
                debug=True, use_flexible_cycles=use_flexible_cycles, draw_all=check_draw_all()))
        else:
            raise ValueError(f"Unknown value {joint_type}")
        links[-1].correct_initial_configuration()
        return links

def draw_multicycles_robot():
    link_length = 5.0
    link_width = 0.5
    pt_start = np.array((0,0))
    pt_end = np.array((link_length,0))

    links = generate_multicycles_robot(LinkDrawing2D, (pt_start, pt_end, link_width), link_length, debug=True)
    links[-1].render(xlim=(-5,10), ylim=(-5,30))
    return links

def main():
    # Test visualization
    links = LinkDrawing2D.generate_sequential_robot(1, 5, ET.Rz() * ET.tx(), [[np.pi/4, 7.5],])
    links[-1].render()

    hex1_links = LinkDrawing2D.make_hexagon_robot()
    hex1_links[-1].render()

    hex2_links = LinkDrawing2D.make_hexagon_robot("P")
    hex2_links[-1].render(xlim=(-20,10), ylim=(-25,5))
    
    _ = draw_multicycles_robot()

    plt.show()

    print("Test passed!")

if __name__ == "__main__":
    main()
