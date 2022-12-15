drawer_pnp_push_commands = [
    ### Task A ###
    {
        "init_pos": [0.5696635276889644, 0.10309521526108646, -0.1203028851440111],
        "drawer_open": True,
        "drawer_yaw": 171.86987153482346,
        "drawer_quadrant": 1,
        "small_object_pos": [0.52500062, -0.16750046, -0.27851104],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 3,
        "command_sequence": [
            ("move_drawer", {}),
            ("move_obj_slide",
                {
                    "target_quadrant": 2,
                }
             )
        ],
        "no_collision_handle_and_small": True,
    },
    ### Task B ###
    {
        "init_pos": [0.8007752866589611, 0.1648154585720787, -0.14012390258012603],
        "init_theta": [180, 0, 90],
        "drawer_open": True,
        "drawer_yaw": 89.60575853084282,
        "drawer_quadrant": 0,
        "small_object_pos": [0.51250274, 0.16758655, -0.26951355],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_drawer", {}),
            ("move_obj_slide",
                {
                    "target_quadrant": 3,
                }
             )
        ],
        "no_collision_handle_and_cylinder": True,
    },
    ### Task C ###
    {
        "init_pos": [0.5024816134266682, -0.07439965024142557, -0.17519156942543912],
        "drawer_open": False,
        "drawer_yaw": 11.04125081594207,
        "drawer_quadrant": 0,
        "small_object_pos": [0.70574489, 0.22969248, -0.35201094],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 1,
        "command_sequence": [
            ("move_obj_slide",
                {
                    "target_quadrant": 2,
                }
             ),
            ("move_drawer", {}),
        ],
        "drawer_hack": True,
    },
]
