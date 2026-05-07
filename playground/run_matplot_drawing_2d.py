# Generating interactive matplots with robots
# Robot Kinematics Playground (Python Library)
# Author: anton.gudym@gmail.com (Anton Gudym)

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

from kinema.elementary_transforms import ET, ETS, SE3

from kinema import LinkDrawing2D
from kinema.link_kinematic import generate_multicycles_robot

class RoboSample(Enum):
    simple = 0
    simple_x4 = 1
    snake = 2
    hexagon = 3
    cycles = 4
class ExampleConstraints(Enum):
    TrackXY = 1
    Parallel_move = 2
    FixedEnd = 3

if __name__ == "__main__" :
    for robot_type in RoboSample:
        # Robot visualization
        link_id_tracking = -1
        constraint_type = ExampleConstraints.Parallel_move #FixedEnd, Parallel_move

        assert constraint_type != ExampleConstraints.FixedEnd or link_id_tracking != -1, "If endpoint is fixed `link_id_tacking` must be changed"

        q_render = None

        if robot_type == RoboSample.simple:
            links = LinkDrawing2D.generate_sequential_robot(1, 5, ET.Rz() * ET.tx(), [[np.pi/4, 7.5],])
        elif robot_type == RoboSample.simple_x4:
            tx = 5.
            links = LinkDrawing2D.generate_sequential_robot(4, tx, ETS(ET.Rz() * ET.tx(tx)))
            q_render = np.array((np.pi/4, -np.pi/4, -np.pi/4, np.pi/4))
        elif robot_type == RoboSample.snake:
            tx = 1
            links = LinkDrawing2D.generate_sequential_robot(10, tx, ETS(ET.Rz() * ET.tx() * ET.tx(tx)),
                                                            draw_all=False, link_width=0.25)
        elif robot_type == RoboSample.hexagon:
            link_id_tracking = -3
            tx = 5.
            jtype = "R"
            #jtype = "P"
            links = LinkDrawing2D.make_hexagon_robot(jtype, use_flexible_cycles=True)
        elif robot_type == RoboSample.cycles:
            link_length = 5.0
            link_width = 0.5
            pt_start = np.array((0,0))
            pt_end = np.array((link_length,0))
            links = generate_multicycles_robot(LinkDrawing2D, (pt_start, pt_end, link_width), link_length, num_cycles=4, debug=True)
        else:
            raise ValueError(f"Unknown robot-type {robot_type}")

        if q_render is None:
            q_render = links[-1].collect_q0()

        if constraint_type == ExampleConstraints.FixedEnd:
            T = links[-1].get_orientation(q_render)
            _ = links[-1].add_constraint(T, (1,1,0,0,0,0))

        if constraint_type == ExampleConstraints.Parallel_move:
            constraint_track = links[link_id_tracking].add_constraint(SE3(), (1,1,0,0,0,1))
        else:
            constraint_track = links[link_id_tracking].add_constraint(SE3(), (1,1,0,0,0,0))

        def robot_control(x: float, y: float, random_init=False):
            global q_render

            T_goal = np.eye(4)
            T_goal[:2, 3] = (x,y)
            constraint_track.update_target_pose(SE3(T_goal))
            
            if random_init:
                q0 = np.random.uniform(-np.pi, np.pi, links[-1].num_joints)
            else:
                q0 = q_render

            try:
                result = links[-1].solve(q0=q0)
                q_render = result.q
                desc = str(result)
            except Exception as e:
                desc = str(e)
            
            links[-1].render(q_render, text_str=desc, xlim=(-20,20), ylim=(-20, 20))

        def mpl_ondrag(event):
            #print(f"{('Single', 'Double')[event.dblclick]} click: button={event.button};" \
            #      f" viewport:({event.x},{event.y});" \
            #      f" coords:({event.xdata}, {event.ydata})", end="\r")
            # 1,2,3 - left,mid,right mouse button
            if  event.xdata is not None and event.ydata is not None and event.button in [1,3] :
                robot_control(event.xdata, event.ydata, event.button == 3 )

        #robot_control(3,3,False)
        links[-1].render(q_render, event_callback=('motion_notify_event', mpl_ondrag), xlim=(-20,20), ylim=(-20, 20))#button_press_event

        plt.show()

