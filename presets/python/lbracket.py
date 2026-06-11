from scene import Scene, Box

scene = Scene("lbracket", 40, 40, 10, suggested_settings="structural")

# Vertical leg (top-right)
scene.add(Box(0, 39, 0, 10, 0, 9, kind="solid"))
# Horizontal leg (bottom)
scene.add(Box(0, 10, 10, 39, 0, 9, kind="solid"))
# Clamp at top of vertical leg
scene.add(Box(0, 10, 39, 39, 0, 9, kind="solid", bc="support", constraint="fix"))
# Load at left tip of horizontal leg
scene.add(Box(39, 39, 0, 0, 4, 5, kind="solid", bc="load", direction=(0, -1, 0)))
