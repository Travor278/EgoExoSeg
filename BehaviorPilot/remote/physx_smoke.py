"""Minimal PhysX GPU smoke test for Isaac Sim 5.1 (BEHAVIOR-1K's pinned version).

Purpose: prove the GPU physics pipeline initializes and steps inside the target
container BEFORE committing to a full OmniGibson install. A conda-pack'd env from
another host previously died here with `no suitable CUDA GPU found` -> SIGSEGV.

Usage:
    python physx_smoke.py            # GPU physics on cuda:0  (the real gate)
    PHYSX_DEVICE=cpu python physx_smoke.py   # CPU fallback, for A/B diagnosis

If GPU segfaults but CPU passes, the container's GPU virtualization is blocking
PhysX's CUDA device enumeration and no amount of reinstalling will fix it.
"""

import os
import sys

DEVICE = os.environ.get("PHYSX_DEVICE", "cuda:0")

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

# Isaac Sim 5.x renamed omni.isaac.core -> isaacsim.core.api. Keep the 4.x path as
# a fallback so a version mismatch fails loudly rather than looking like a PhysX bug.
try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid

    API = "isaacsim.core.api (5.x)"
except ImportError:
    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicCuboid

    API = "omni.isaac.core (4.x)"

import numpy as np  # noqa: E402

print(f"PHYSX-SMOKE api={API} device={DEVICE}", flush=True)

world = World(physics_dt=1 / 60, rendering_dt=1 / 60, backend="torch", device=DEVICE)
world.scene.add_default_ground_plane()
cube = world.scene.add(
    DynamicCuboid(prim_path="/World/cube", position=np.array([0.0, 0.0, 2.0]), size=0.5)
)
world.reset()

z0 = float(cube.get_world_pose()[0][2])
print(f"PHYSX-SMOKE reset ok, z0={z0:.3f}", flush=True)

for _ in range(120):
    world.step(render=False)  # pure physics stepping — this is where it crashed before

z1 = float(cube.get_world_pose()[0][2])
print(f"PHYSX-SMOKE cube z {z0:.3f} -> {z1:.3f} (should fall)", flush=True)

if z1 >= z0 - 0.5:
    print("PHYSX-FAIL cube did not fall — physics not stepping", flush=True)
    app.close()
    sys.exit(2)

print("PHYSX-OK", flush=True)
app.close()
