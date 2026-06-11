from scene import Scene, Box


scene = Scene("drone_frame", 40, 40, 20, symmetry="xy",
              suggested_settings="drone")

scene.domain(0, 39, 0, 39, 0, 19)

engine_height = 15
body_height = 10

# central body
scene.add(Box(15, 24, 15, 24, body_height, body_height, kind="fixed_solid", bc="support", constraint="fix"))
# clearance above body for payload
scene.add(Box(16, 23, 16, 23, body_height+1, 19, kind="fixed_void"))

# engine case 1 : vertical
scene.add(Box(0, 4, 0, 4, engine_height, engine_height, kind="fixed_solid", bc="load", direction=(0, 0, -1), case=0))
scene.add(Box(35, 39, 0, 4, engine_height, engine_height, kind="fixed_solid", bc="load", direction=(0, 0, -1), case=1))
scene.add(Box(0, 4, 35, 39, engine_height, engine_height, kind="fixed_solid", bc="load", direction=(0, 0, -1), case=2))
scene.add(Box(35, 39, 35, 39, engine_height, engine_height, kind="fixed_solid", bc="load", direction=(0, 0, -1), case=3))

# engine case 2 : X axis
scene.add(Box(0, 4, 0, 4, engine_height, engine_height, kind="fixed_solid", bc="load", direction=(1, 0, -1), case=4))
scene.add(Box(35, 39, 0, 4, engine_height, engine_height, kind="fixed_solid", bc="load", direction=(-1, 0, -1), case=5))
scene.add(Box(0, 4, 35, 39, engine_height, engine_height, kind="fixed_solid", bc="load", direction=(1, 0, -1), case=6))
scene.add(Box(35, 39, 35, 39, engine_height, engine_height, kind="fixed_solid", bc="load", direction=(-1, 0, -1), case=7))

# engine case 3 : Y axis
scene.add(Box(0, 4, 0, 4, engine_height, engine_height, kind="fixed_solid", bc="load", direction=(0, 1, -1), case=8))
scene.add(Box(35, 39, 0, 4, engine_height, engine_height, kind="fixed_solid", bc="load", direction=(0, 1, -1), case=9))
scene.add(Box(0, 4, 35, 39, engine_height, engine_height, kind="fixed_solid", bc="load", direction=(0, -1, -1), case=10))
scene.add(Box(35, 39, 35, 39, engine_height, engine_height, kind="fixed_solid", bc="load", direction=(0, -1, -1), case=11))