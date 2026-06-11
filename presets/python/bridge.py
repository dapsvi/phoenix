from scene import Scene, Box

LENGTH = 150

scene = Scene("bridge", LENGTH, 50, 10, symmetry="xz", suggested_settings="bridge_settings")

scene.add(Box(0, LENGTH - 1,            0, 23,          3, 6, kind="solid"))

scene.add(Box(0, 0,                     0, 0,           3, 6, kind="fixed_solid", bc="support", constraint="fix"))
scene.add(Box(LENGTH - 1, LENGTH - 1,   0, 0,           3, 6, kind="fixed_solid", bc="support", constraint="fix"))
scene.add(Box(0, 0,                     23, 23,         3, 6, kind="fixed_solid", bc="support", constraint="fix_z"))
scene.add(Box(LENGTH - 1, LENGTH - 1,   20, 10,         3, 6, kind="fixed_solid", bc="support", constraint="fix_z"))

scene.add(Box(0, LENGTH - 1,            23, 23,         0, 9, kind="fixed_solid", bc="load", direction=(0, -1, 0)))
scene.add(Box(0, LENGTH - 1,            24, 49,         1, 8, kind="fixed_void"))