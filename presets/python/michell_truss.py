from scene import Scene, Box

scene = Scene("michell_truss", 32, 20, 12, suggested_settings="structural")

scene.add(Box(0, 31, 0, 19, 0, 11, kind="solid"))
scene.add(Box(0, 0, 0, 19, 0, 11, kind="solid", bc="support", constraint="fix"))
scene.add(Box(31, 31, 0, 0, 5, 6, kind="solid", bc="load", direction=(0, -1, 0)))
