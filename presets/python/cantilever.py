from scene import Scene, Box

scene = Scene(
    "cantilever", 40, 20, 20,
    symmetry="z",
    suggested_settings="structural",
)
scene.add(Box(0, 39, 0, 19, 0, 19, kind="solid"))
scene.add(Box(0, 0, 0, 19, 0, 19, kind="solid", bc="support", constraint="fix"))
scene.add(Box(39, 39, 0, 0, 10, 10, kind="solid", bc="load", direction=(0, -1, 0)))
