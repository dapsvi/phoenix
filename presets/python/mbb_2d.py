from scene import Scene, Box

L, Hgt = 60, 20

scene = Scene("mbb_2d", nx=L, ny=Hgt, nz=1,
              symmetry=None,
              suggested_settings="bench_2d")

scene.add(Box(0, L-1, 0, Hgt-1, 0, 0, kind="solid"))

scene.add(Box(0, 0, 0, Hgt-1, 0, 0,
              kind="solid", bc="support", constraint="fix_x"))

# Roller (bottom-right corner): lock vertical (Y) motion only
scene.add(Box(L-1, L-1, 0, 0, 0, 0,
              kind="solid", bc="support", constraint="fix_y"))

# Load: one downward point load at the top-left corner
scene.add(Box(0, 0, Hgt-1, Hgt-1, 0, 0,
              kind="solid", bc="load", direction=(0, -1, 0)))