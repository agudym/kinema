import numpy as np
import matplotlib.pyplot as plt

from kinema.elementary_transforms import ET
from kinema.link_drawing_2d import LinkDrawing2D, draw_multicycles_robot

def test_link_drawing_2d_basic():
    links = LinkDrawing2D.generate_sequential_robot(1, 5, ET.Rz() * ET.tx(), [[np.pi/4, 7.5],])
    assert len(links) > 0

    hex1_links = LinkDrawing2D.make_hexagon_robot()
    assert len(hex1_links) > 0

    hex2_links = LinkDrawing2D.make_hexagon_robot("P")
    assert len(hex2_links) > 0
    
    links = draw_multicycles_robot()
    assert len(links) > 0
