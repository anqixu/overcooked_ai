from overcooked_ai.ai_types import map_from_short_label_to_object_id


# NOTE: all params tuned for Overcooked 1-1
game_board_tl_y = 90
game_board_tl_x = 100
game_board_br_y = 900
game_board_br_x = 1750

num_pixels_per_tile = 100


world_1_1_tile_short_labels = (
    (
        "CtrTp",
        "CtrTp",
        "CtrTp",
        "CtrTp",
        "CtrTp",
        "",
        "",
        "",
        "CtrTp",
        "CtrTp",
        "CtrTp",
        "CtrTp",
        "CtrTp",
    ),
    ("CtrTp", "", "", "", "", "", "", "", "", "", "", "", "CtrTp"),
    ("TrBin", "", "", "", "", "", "", "", "", "", "", "", "TPass"),
    ("CtrTp", "", "", "", "", "", "", "", "", "", "", "", "BPass"),
    ("TnaCt", "", "", "CtrTp", "CtrTp", "", "", "", "CtrTp", "CtrTp", "", "", "PlRtn"),
    ("CtrTp", "", "", "", "", "", "", "", "", "", "", "", "ShrCt"),
    ("CtrTp", "", "", "", "", "", "", "", "", "", "", "", "CtrTp"),
    ("CtrTp", "CtBrd", "CtBrd", "", "", "", "", "", "", "", "CtBrd", "CtBrd", "CtrTp"),
)


world_1_1_tile_object_ids = tuple(
    tuple(map_from_short_label_to_object_id[label] for label in row)
    for row in world_1_1_tile_short_labels
)
