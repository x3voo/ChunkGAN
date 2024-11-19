import amulet

air = amulet.Block('universal_minecraft', 'air')
air_replacements = [
    "snow", "melon", "double_plant", "plant", "pumpkin",
    "carved_pumpkin", "jack_o_lantern", "hay_block", "bee_nest", "honeycomb_block",
    "moss_block",  "moss_carpet",# problematyczne te dwa
     "mushroom_stem", "brown_mushroom_block", "red_mushroom_block",
    "ochre_froglight", "verdant_froglight", 
     "glowstone", "water", "ice", "packed_ice",
    "blue_ice", "coral_block", "seagrass", "prismarine", 'log', 'leaves', "mangrove_roots",
    "muddy_mangrove_roots",
    
    "lava", "sugar_cane", "bubble_column", 'red_mushroom', 'brown_mushroom'
]
# 24.5 -22.8 24,3

stone = amulet.Block('universal_minecraft', 'stone')
stone_replacements = [
    "blackstone", "basalt", "smooth_basalt", "end_stone", "amethyst_block",
    "sculk", "calcite", "dripstone_block", "magma_block",
    "obsidian", "crying_obsidian", "netherrack", "crimson_nylium", "warped_nylium", "emerald_ore",
    "deepslate_emerald_ore", "lapis_ore", "deepslate_lapis_ore", "nether_gold_ore", "nether_quartz_ore",
    "ancient_debris", "deepslate", "redstone_ore", "deepslate_redstone_ore", 'diorite', 'tuff', 'granite',
    'andesite', 'gravel',
    'coal_ore', 'deepslate_coal_ore',
    'copper_ore', 'deepslate_copper_ore', 'raw_copper_block',
    'iron_ore', 'deepslate_iron_ore', 'raw_iron_block',
    'gold_ore', 'deepslate_gold_ore', 'raw_gold_block',
    'diamond_ore', 'deepslate_diamond_ore',
    #hmm problem, jednak wygenerowalo jakies struktury
    'mossy_cobblestone', 'cobblestone'
]


cave_air = amulet.Block('universal_minecraft', 'cave_air')
cave_air_replacements = [ "budding_amethyst", 'sculk_vein', "sculk_catalyst", "small_amethyst_bud", "medium_amethyst_bud",
                          "large_amethyst_bud", "amethyst_cluster", "cobweb", "pointed_dripstone", "bone_block",
                          "glow_lichen",
                          'spawner', 'chest', 
    ]

dirt = amulet.Block('universal_minecraft', 'dirt')
dirt_replacements = [
    "snow_block", "coarse_dirt", "rooted_dirt", "farmland", "mud", "clay",
    "soul_soil", "nether_wart_block", "warped_wart_block", 'grass_block',
    "podzol", "mycelium", "grass_path"
]

bedrock = amulet.Block('universal_minecraft', 'bedrock')

sand = amulet.Block('universal_minecraft', 'sand')
sand_replacements = ["sandstone", "red_sand", "red_sandstone", "soul_sand"]

air_set = set(air_replacements)
stone_set = set(stone_replacements)
cave_air_set = set(cave_air_replacements)
dirt_set = set(dirt_replacements)
sand_set = set(sand_replacements)

block_height_layer_rank = [air, dirt, sand, stone, cave_air, bedrock]

# Modify existing pallet to use reduced amount of block, in a sense loselly
# compressing information. 
def merge_palettes(block_palette):
    new_palette = []
    for target_index, target_block in enumerate(block_palette):
        if target_block.base_name == air.base_name:
            new_palette.append(air)
        elif target_block.base_name == stone.base_name:
            new_palette.append(stone)
        elif target_block.base_name == cave_air.base_name:
            new_palette.append(cave_air)
        elif target_block.base_name == dirt.base_name:
            new_palette.append(dirt)
        elif target_block.base_name == sand.base_name:
            new_palette.append(sand)
        elif target_block.base_name == bedrock.base_name:
            new_palette.append(bedrock)

        elif target_block.base_name in air_set:
            new_palette.append(air)
        elif target_block.base_name in stone_set:
            new_palette.append(stone)
        elif target_block.base_name in cave_air_set:
            new_palette.append(cave_air)
        elif target_block.base_name in dirt_set:
            new_palette.append(dirt)
        elif target_block.base_name in sand_set:
            new_palette.append(sand)

        else:
            print(f"[Warn] [custom_block_palette] not found {target_block}!")
            new_palette.append(air)

    return new_palette
