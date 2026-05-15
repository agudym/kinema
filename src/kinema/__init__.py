import jax
devices = jax.devices()
print(f"JAX Devices available: {devices}")
print(f"JAX Default Device: {devices[0]}")

from .link_kinematic import LinkKinematic
from .link_connected import LinkConnected
from .link_drawing_2d import LinkDrawing2D
from .link_blender_3d import LinkBlenderDrawing3D, create_hexapod, create_panda, create_robopro_rc10

