import sys
import os
# Amulet takes to much time to load on Colab which costs $$$
if 'idlelib.run' in sys.modules:
    import amulet
    import custom_block_palette_v1 as CBP
    
import numpy as np
import time
import gc
from PIL import Image
import pickle
import shutil
import datetime

GAME_HEIGHT_LIMIT = 320     # -64..319
GAME_GEN_HIEGHT_LIMIT = 256 # -64..255
GAME_SEA_LEVEL = 62         # Sea water level is 62 (62..XX..63)
GAME_BOTTOM_LAYER = -64     # Last layer is -64

CHUNK_SIZE = 16             # 0..15

ABS_GAME_HEIGHT_LIMIT = abs(GAME_HEIGHT_LIMIT) + abs(GAME_BOTTOM_LAYER) #384
ABS_GAME_GEN_HEIGHT_LIMIT = abs(GAME_GEN_HIEGHT_LIMIT) + abs(GAME_BOTTOM_LAYER) #320

ABS_SEA_LEVEL = abs(GAME_SEA_LEVEL) + abs(GAME_BOTTOM_LAYER)

# IN-GAME VALUES
INPUT_HEIGHTS = [
        GAME_GEN_HIEGHT_LIMIT,  # Naturally-generated terrain cap
        256                     # Plains-only-320x320 highest block, there is actauly a block at the limit of generator.... at 255
    ]

INPUT_HEIGHT_GAME = INPUT_HEIGHTS[0]
INPUT_HEIGHT = INPUT_HEIGHTS[0] + abs(GAME_BOTTOM_LAYER)


# [TODO] FIX HEIGHT LIMITS

#INPUT_HEIGHT
#INPUT_HEIGHT_GAME, ABS_SEA_LEVEL

class Block():
    def __init__(self, namespace, base_name):
        self.base_name = base_name
        self.namespace = namespace
    def __repr__(self):
        return f"LocalBlock({self.namespace}:{self.base_name})"

class MCReader():
    def __init__(self, world_size = 1, height_limit = INPUT_HEIGHT, chunk_size = CHUNK_SIZE):
        self.Wx = world_size
        self.Wz = world_size
        self.Cx = chunk_size
        self.Cy = height_limit
        self.Cz = chunk_size

        chunk_size_in_mb = round(world_size * world_size * chunk_size * chunk_size * height_limit / 1000000, 2)
        print(f"Estimated size: {chunk_size_in_mb} MB (thats also a minimum requirement of free RAM)")

        self.chunks = np.zeros((self.Wx, self.Wz, self.Cx, self.Cy, self.Cz), dtype=np.uint8)
        self.block_palette = None
        self.block_palette_amulet = None
        self.heightmaps = np.zeros((self.Wx, self.Wz, self.Cx, self.Cz), dtype=np.uint16)
        self.cave_densities = np.zeros((self.Wx, self.Wz), dtype=np.uint16)

        #self.collapsed_chunks =

    def conv_palette_to_local(self):
        self.block_palette = [
            Block('universal_minecraft', self.block_palette_amulet[0].base_name),
            Block('universal_minecraft', self.block_palette_amulet[1].base_name),
            Block('universal_minecraft', self.block_palette_amulet[2].base_name),
            Block('universal_minecraft', self.block_palette_amulet[3].base_name),
            Block('universal_minecraft', self.block_palette_amulet[4].base_name),
            Block('universal_minecraft', self.block_palette_amulet[5].base_name),
        ]
        return True
    def conv_palette_to_amulet(self):
        self.block_palette_amulet = [
            amulet.Block('universal_minecraft', self.block_palette[0].base_name),
            amulet.Block('universal_minecraft', self.block_palette[1].base_name),
            amulet.Block('universal_minecraft', self.block_palette[2].base_name),
            amulet.Block('universal_minecraft', self.block_palette[3].base_name),
            amulet.Block('universal_minecraft', self.block_palette[4].base_name),
            amulet.Block('universal_minecraft', self.block_palette[5].base_name),
        ]
        return True

    # [102400] 320x320 = 8388.61 MB ('Single Biome World - Plains' MAX) | Extreme fine-tunning
    # [65536]  256x256 = 5368.71 MB                                     | Fine-tunning
    # [16384]  128x128 = 1342.18 MB                                     | Training
    # [4096]    64x64  = 335.54 MB                                      | Training/Testing      | 12 min + 8 min
    # [1024]    32x32  = 83.89 MB                                       | Testing
    def save(self, filename):
        
        np.save(f"{filename}_{self.Wx}x{self.Wz}_chunks.npy", self.chunks)
        np.save(f"{filename}_{self.Wx}x{self.Wz}_heightmaps.npy", self.heightmaps)
        np.save(f"{filename}_{self.Wx}x{self.Wz}_cave_densities.npy", self.cave_densities)

        # Save the smaller attributes using pickle
        save_data = {
            'block_palette': self.block_palette,
            'Wx': self.Wx,
            'Wz': self.Wz,
            'Cx': self.Cx,
            'Cy': self.Cy,
            'Cz': self.Cz
        }
    
        with open(f"{filename}_{self.Wx}x{self.Wz}_meta.pkl", 'wb') as file:
            pickle.dump(save_data, file)
            
        return True

    def load(self, filename):
        # Load large numpy arrays
        self.chunks = np.load(f"{filename}_{self.Wx}x{self.Wz}_chunks.npy")
        self.heightmaps = np.load(f"{filename}_{self.Wx}x{self.Wz}_heightmaps.npy")
        self.cave_densities = np.load(f"{filename}_{self.Wx}x{self.Wz}_cave_densities.npy")

        # Load the other attributes from the pickle file
        with open(f"{filename}_{self.Wx}x{self.Wz}_meta.pkl", 'rb') as file:
            save_data = pickle.load(file)
    
        # Restore attributes
        self.block_palette = save_data['block_palette']
        self.Wx = save_data['Wx']
        self.Wz = save_data['Wz']
        self.Cx = save_data['Cx']
        self.Cy = save_data['Cy']
        self.Cz = save_data['Cz']
        
        return True

    # Read world from game save file, applies fixes and tranlates to IDs using custom palette.
    # Dimensions of dumped world are set by this class.
    def dump_game_save_file(self, save_path):
        log_head = "[MCReader] [dump_game_save_file]" 
        print(f"[Info] {log_head} Save file dumping started.")
        # uint8 might be unsafe!
        chunks_buffer = np.zeros((self.Wz, self.Cx, self.Cy, self.Cz), dtype=np.uint16) # uint16 is faster somehow

        world = amulet.load_level(save_path)
        
        air_block_id = world.block_palette.get_add_block(CBP.air)
        cave_air_block_id = world.block_palette.get_add_block(CBP.cave_air)
        
        for Wx in range(self.Wx):
            print(f"Wx: {Wx}")
            start_time = time.time()

            for Wz in range(self.Wz):
                chunks_buffer[Wz] = world.get_chunk(Wx, Wz, "minecraft:overworld").blocks[:, -64:INPUT_HEIGHT_GAME, :]

                # Caves Fix
                #air_block_id = world.block_palette.get_add_block(CBP.air)
                #cave_air_block_id = world.block_palette.get_add_block(CBP.cave_air)
                for x in range(self.Cx):
                    for z in range(self.Cz):
                        for y in range(1, ABS_SEA_LEVEL + 1): # Skip first, as it never contains air blocks
                            if chunks_buffer[Wz][x, y, z] == air_block_id:
                                chunks_buffer[Wz][x, y, z] = cave_air_block_id

            # Prepare original palette and create new palette
            marged_palette = CBP.merge_palettes(world.block_palette)
            new_palette = CBP.block_height_layer_rank # specific indexing order

            # Set new_palette as new default palette
            self.block_palette = new_palette

            # Create translation palette
            translator = [None] * len(marged_palette)

            for idx1, t1 in enumerate(marged_palette):
                found = False
                for idx2, t2 in enumerate(new_palette):
                    if t1.base_name == t2.base_name:
                        translator[idx1] = idx2
                        found = True
                        break
                if found == False:
                    print('[ERROR] {log_head} {t1.base_name} does not exist in custom palette')

            # Soft encode
            for Wz in range(self.Wz):
                for x in range(self.Cx):
                    for z in range(self.Cz):
                        for y in range(self.Cy):
                            mc_id = chunks_buffer[Wz][x, y, z]
                            nn_id = translator[mc_id]
                            self.chunks[Wx][Wz][x, y, z] = nn_id

            

            # Purge RAM cache
            world.chunks.unload()
            
            # Check progress
            end_time = time.time()
            time_left = (end_time - start_time) * (self.Wx - Wx)
            print(f'[Info] {log_head} Execution time: ', (end_time - start_time))
            mm, ss = divmod(time_left, 60)
            hh, mm = divmod(mm, 60)
            print(f'[Info] {log_head} Estimated time left:', hh, 'Hours', mm, 'Minutes', ss, 'Seconds')

        # Free mem
        world.close()
        del world
        gc.collect()

        self.block_palette_amulet = self.block_palette
        self.conv_palette_to_local()

        gc.collect()
        print("[Info] {log_head} Finished dumping save file.")
        return True

    def get_block_id(self, block_name):
        log_head = "[MCReader] [get_block_id]" 
        for idx, t in enumerate(self.block_palette):
            if t.base_name == block_name:
                return idx
        print(f"[ERROR] {log_head} Couldn't find {block_name} in self.block_palette! Returning 0 instead.")
        return 0

    def get_block(self, id_or_name):
        log_head = "[MCReader] [get_block]" 
        if type(id_or_name) == type("string"):
            for idx, t in enumerate(self.block_palette):
                if t.base_name == id_or_name:
                    return t
            print(f"[ERROR] {log_head} Couldn't find block of name {id_or_name} in self.block_palette! Returning air block instead.")
            return CBP.air
        elif type(id_or_name) == type(int(1)):
            for idx, t in enumerate(self.block_palette):
                if idx == id_or_name:
                    return t
            print(f"[ERROR] {log_head} Couldn't find block of id {id_or_name} in self.block_palette! Returning air block instead.")
            return CBP.air
        else:
            print(f"[ERROR] {log_head} Got {type(id_or_name)} type, expected {type('string')} or {type(int(1))}. Returning air block instead.")
            return CBP.air
            

    def create_heightmaps(self):
        log_head = "[MCReader] [create_heightmaps]"
        print(f"[Info] {log_head} Creating heightmaps...")
        air_id = self.get_block_id("air")
        cave_air_id = self.get_block_id("cave_air")
        for Wx in range(self.Wx):
            print(f"Wx: {Wx}")
            start_time = time.time()
            
            for Wz in range(self.Wz):
                for x in range(self.Cx):
                    for z in range(self.Cz):
                        for y in range(self.Cy - 1, 0, -1):
                            if self.chunks[Wx][Wz][x, y, z] != air_id:
                                if self.chunks[Wx][Wz][x, y, z] != cave_air_id:
                                    self.heightmaps[Wx][Wz][x, z] = y
                                    break

            # Check progress
            end_time = time.time()
            time_left = (end_time - start_time) * (self.Wx - Wx)
            print(f'[Info] {log_head} Execution time: ', (end_time - start_time))
            mm, ss = divmod(time_left, 60)
            hh, mm = divmod(mm, 60)
            print(f'[Info] {log_head} Estimated time left:', hh, 'Hours', mm, 'Minutes', ss, 'Seconds')
            
        gc.collect()
        print("[Info] {log_head} Finished creating heightmaps.")
        return True

    # It is not really correct, as caves appear above sea level, so it is more like "air pockets under sea level"
    def calculate_caves_densities(self):
        log_head = "[MCReader] [calculate_caves_densities]"
        print(f"[Info] {log_head} Calculating caves densities.")
        cave_air_id = self.get_block_id("cave_air")
        for Wx in range(self.Wx):
            print(f"Wx: {Wx}")
            start_time = time.time()
            
            for Wz in range(self.Wz):
                self.cave_densities[Wx][Wz] = np.count_nonzero(self.chunks[Wx][Wz][:, 1:ABS_SEA_LEVEL-1 ,:] == cave_air_id)

            # Check progress
            end_time = time.time()
            time_left = (end_time - start_time) * (self.Wx - Wx)
            print(f'[Info] {log_head} Execution time: ', (end_time - start_time))
            mm, ss = divmod(time_left, 60)
            hh, mm = divmod(mm, 60)
            print(f'[Info] {log_head} Estimated time left:', hh, 'Hours', mm, 'Minutes', ss, 'Seconds')
            
        gc.collect()
        print("[Info] {log_head} Finished calculating caves densities.")
        return True

    def get_neighbours(self, idx, dropout = 0):
        #if CHUNKS[idx]

        # [TODO] on the edges might not find some sides, check for that
        # dropout, just randomly drop from 0 to 4 sides on output (for training, for the model those inputs are optional thats why)
        
        WEST_SIDE = self.cave_densities[Wx, Wz][15, :, :]
        EAST_SIDE = self.cave_densities[Wx, Wz][0, :, :]
        NORTH_SIDE = self.cave_densities[Wx, Wz][:, :, 15]
        SOUTH_SIDE = self.cave_densities[Wx, Wz][:, :, 0]
        return []

    # [TODO]
    def chunk_info(self, Wx, Wz):
        chunk_size = self.Cx * self.Cy * self.Cz
        print(f"Chunk [{Wx}, {Wz}]")
        blocks = np.unique(self.chunks[Wx, Wz])
        print(f" Unique blocks {blocks}")
        print(" Blocks:")
        for idx, t in enumerate(self.block_palette):
            count = np.count_nonzero(self.chunks[Wx, Wz] == idx)
            percentage = round(100 * count / chunk_size, 2)
            print(f"  [{idx}] '{t.base_name}' : {count} ({percentage} %)")
        print(f" Caves density: {self.cave_densities[Wx][Wz]} ({round((100 * self.cave_densities[Wx, Wz] / chunk_size), 2)} %)")
        print(" Elevation:")
        highest_point = np.max(self.heightmaps[Wx][Wz])
        lowest_point = np.min(self.heightmaps[Wx][Wz])
        print(f"  Highest point ({highest_point - (64+62)})")
        print(f"  Lowest point ({lowest_point - (64+62)})")
        print(" Radius:")
        print(f"  Highest point ({highest_point}) (In-game: {highest_point-64})")
        print(f"  Lowest point ({lowest_point}) (In-game: {lowest_point-64})")
        print(" Neighbours:")
        print(f"  [TODO]")
        print(" Heightmap (In-game):")
        print(self.heightmaps[Wx][Wz]-64) # [TODO] cast to int not uint
        print(" Generating heightmap image...")
        img = self.get_chunks_heightmap_as_bitmap(Wx, Wz, 32)
        img.show()

    def create_heightmap(self, chunk):
        heightmap = np.zeros((self.Cx, self.Cz), dtype=np.int16)
        
        air_id = self.get_block_id("air")
        cave_air_id = self.get_block_id("cave_air")
        for x in range(self.Cx):
            for z in range(self.Cz):
                for y in range(self.Cy - 1, 0, -1):
                   if chunk[x, y, z] != air_id:
                       if chunk[x, y, z] != cave_air_id:
                           heightmap[x, z] = y
                           break
        return heightmap

    def chunk_info2(self, encoded_chunk):
        chunk = self.decode(encoded_chunk)
        heightmap = self.create_heightmap(chunk)

        cave_air_id = self.get_block_id("cave_air")
        cave_densities = np.count_nonzero(chunk[:, 1:ABS_SEA_LEVEL-1 ,:] == cave_air_id)
        
        chunk_size = 16 * 320 * 16
        print(f"[Chunk info]")
        blocks = np.unique(chunk)
        print(f" Unique blocks {blocks}")
        print(" Blocks:")
        palette = ['air', 'dirt', 'sand', 'stone', 'cave_air', 'bedrock']
        for idx, t in enumerate(palette):
            count = np.count_nonzero(chunk == idx)
            percentage = round(100 * count / chunk_size, 2)
            print(f"  [{idx}] '{t}' : {count} ({percentage} %)")
        print(f" Caves density: {cave_densities} ({round((100 * cave_densities / chunk_size), 2)} %)")
        print(" Elevation:")
        highest_point = np.max(heightmap)
        lowest_point = np.min(heightmap)
        print(f"  Highest point ({highest_point - (64+62)})")
        print(f"  Lowest point ({lowest_point - (64+62)})")
        print(" Radius:")
        print(f"  Highest point ({highest_point}) (In-game: {highest_point-64})")
        print(f"  Lowest point ({lowest_point}) (In-game: {lowest_point-64})")
        print(" Neighbours:")
        print(f"  [TODO]")
        print(" Heightmap (In-game):")
        print(heightmap-64)


    # ------UTILS------
    def get_chunks_heightmap_as_bitmap(self, Wx, Wz, scale_factor = 4):
        normalized_bitmap = (self.heightmaps[Wx, Wz] * 255 / self.Cy).astype(np.uint8)
        #print(normalized_bitmap)
        image = Image.fromarray(normalized_bitmap, mode='L')
        new_size = (image.width * scale_factor, image.height * scale_factor)
        scaled_image = image.resize(new_size, resample=Image.NEAREST) # avoid interpolation
        return scaled_image

    def find_highest_block(self, world_size, save_path):
        log_head = "[MCReader] [find_highest_block]"
        # assuming air has id 0, its bad but fast
        
        # world.chunks.purge # clears chunks from RAM and cache
        # world.chunks.unload # clears only from RAM
        
        chunk_slice = np.zeros((self.Cx, 1, self.Cz), dtype=int)
        
        world = amulet.load_level(save_path)
        
        for Y_SLICE in range(GAME_GEN_HIEGHT_LIMIT, GAME_SEA_LEVEL, -1):    
            for Wx in range(int(512/16), int(2560/16)):
                print(f"Y_SLICE[{Y_SLICE}] - Wx:{Wx}")
                start_time = time.time()
                
                for Wz in range(int(2048/16),int(3584/16)):
                    #is_there_any_non_air_block = np.sum(world.get_chunk(Wx, Wz, "minecraft:overworld").blocks[:, Y_SLICE:Y_SLICE+1, :])
                    chunk_slice = world.get_chunk(Wx, Wz, "minecraft:overworld").blocks[:, Y_SLICE, :]
                    for x in range(16):
                        for z in range(16):       
                            #if world.get_chunk(Wx, Wz, "minecraft:overworld").blocks[x, Y_SLICE, z] != 0:
                            if chunk_slice[x, 0, z] != 0:
                                print(f"Found highest block in chunk [{Wx}][{Wz}] at Y = {Y_SLICE}")
                                return [Wx, Wz]

                # Check progress
                end_time = time.time()
                time_left = (end_time - start_time) * (world_size - Wx)
                print(f'[Info] {log_head} Execution time: ', (end_time - start_time))
                mm, ss = divmod(time_left, 60)
                hh, mm = divmod(mm, 60)
                print(f'[Info] {log_head} Estimated time left:', hh, 'Hours', mm, 'Minutes', ss, 'Seconds')

            world.chunks.unload()
          
        # Free mem
        world.close()
        del world
        gc.collect()
        
        return GAME_SEA_LEVEL

    # Decode chunk from one-hot encoding back to IDs
    def decode(self, chunk):
        decoded_chunk = np.zeros((self.Cx, self.Cy, self.Cz), dtype=np.uint8)
        for x in range(self.Cx):
            for z in range(self.Cz):
                for y in range(self.Cy):
                    index_of_max = np.argmax(chunk[x,y,z])
                    decoded_chunk[x,y,z] = index_of_max
        return decoded_chunk

    def fast(self):
        res1 = np.load("res1.npy")
        res2 = self.decode(res1)
        print(np.unique(res2))
        txt = self.generate_save_file(self.chunks[0,0], res2, "NewModel")
        self.import_save_to_minecraft(txt)  

    # ------MINECRAFT------
    
    def overwrite_chunk(self, world, x, z, chunk, decorate = False):
        # Create new chunk object for coords x,z
        new_chunk = amulet.api.chunk.Chunk(x, z)

        grass = amulet.Block('universal_minecraft', 'grass_block')
        water = amulet.Block('universal_minecraft', 'water')

        if self.block_palette_amulet == None:
            self.conv_palette_to_amulet()

        # Fill first XXX layers (-64, XXX)
        for x in range(CHUNK_SIZE):
            for z in range(CHUNK_SIZE):
                for y in range(-64, INPUT_HEIGHT_GAME):
                    new_chunk.blocks[x, y, z] = world.block_palette.get_add_block(self.block_palette_amulet[chunk[x][y+64][z]])

        if decorate == True:
            # GRASS
            for x in range(CHUNK_SIZE):
                for z in range(CHUNK_SIZE):
                    for y in range(INPUT_HEIGHT_GAME-1, GAME_SEA_LEVEL - 1, -1):        
                        if chunk[x][y+64][z] != 0:
                            if chunk[x][y+64][z] == 1:
                                new_chunk.blocks[x, y, z] = world.block_palette.get_add_block(grass)
                                break

            # WATER
            for x in range(CHUNK_SIZE):
                for z in range(CHUNK_SIZE):
                    for y in range(-64, GAME_SEA_LEVEL + 1):        
                        if chunk[x][y+64][z] == 0:
                            new_chunk.blocks[x, y, z] = world.block_palette.get_add_block(water)

        # Override world with "new_chunk"
        world.put_chunk(new_chunk, "minecraft:overworld")
        new_chunk.changed = True
        
    def create_new_mc_save(self, name = "AI"):
        isFile = os.path.isfile("Blank World\level.dat")
        if isFile:
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
            new_save_name = f"{name}-{timestamp}"
            new_save_path = os.path.join(new_save_name)
            os.makedirs(new_save_path, exist_ok=True)
        
            # Copy the contents of the "Blank World" directory to the new save directory
            shutil.copytree("Blank World", new_save_path, dirs_exist_ok=True)
        
            # Return the path to the new save
            return new_save_path
        else:
            raise FileNotFoundError("The template world 'Blank World' does not exist.")

    # Creates save file and write chunks
    def generate_save_file(self, input, prediction, name = "AI"):
         # Create a new Minecraft save and get its path
        path = self.create_new_mc_save(name)

        # Load the newly "created" world using the Amulet API
        new_world = amulet.load_level(path)

         # Create and overwrite chunks based on the input and prediction
        self.overwrite_chunk(new_world, 0, 0, input)
        self.overwrite_chunk(new_world, 1, 0, prediction)

        # Save and close the new world
        new_world.save()
        new_world.close()

        print(f"Generated new save at {path}.")
        return path

    def generate_save_file_2(self, input, name = "AI", decorate = True):
        path = self.create_new_mc_save(name)

        new_world = amulet.load_level(path)

        for x in range(np.shape(input)[0]):
            for y in range(np.shape(input)[1]):
                self.overwrite_chunk(new_world, x, y, input[x,y], decorate)

        new_world.save()
        new_world.close()

        print(f"Generated new save at {path}.")
        return path

    # Import generated save to the game (just copies)
    def import_save_to_minecraft(self, save):
        game_profile_path = os.path.join("minecraft-game", "minecraft1.20.1-forge")
        save_path = os.path.join(game_profile_path, "saves")

        # Ensure the destination save path exists
        os.makedirs(save_path, exist_ok=True)

        # Define the full destination path for the new save
        destination_path = os.path.join(save_path, os.path.basename(save))

        # Copy the entire save folder to the destination path
        shutil.copytree(save, destination_path, dirs_exist_ok=True)
    
        print(f"Imported {save} to {save_path}.")


    # ------AI-DATA------
    def get_ml_input(self, Wx, Wz):
        heightmap = self.heightmaps[Wx, Wz] # input
        caves = self.cave_densities[Wx, Wz] # input
        return [heightmap, caves]

    def get_ml_input_set(self, offset, amount):
        dataset_input_heightmaps = np.empty((amount, 16, 16), dtype=np.uint16)
        dataset_input_caves = np.empty((amount, 1), dtype=np.uint16)
        dataset_output = np.empty((amount, 16, 320, 16), dtype=np.uint8)
        i = 0
        for QQ in range(offset, offset + amount):
            Wx = QQ % self.Wx
            Wz = QQ // self.Wz
            dataset_input_heightmaps[i] = self.heightmaps[Wx, Wz]
            dataset_input_caves[i] = self.cave_densities[Wx, Wz]
            dataset_output[i] = self.chunks[Wx, Wz]
            i = i + 1

        stone_id = self.get_block_id("stone")
        air_id = self.get_block_id("air")
        cave_air_id = self.get_block_id("cave_air")
        
        # Pre-defined chunk volume
        dataset_input_volume = np.zeros(shape=(amount, 16, 320, 16), dtype=np.uint8)
        for d in range(amount):
            for x in range(self.Cx):
                for z in range(self.Cz):
                    dataset_input_volume[d][x, 0:int(dataset_input_heightmaps[d][x, z]),z] = stone_id
                    dataset_input_volume[d][x, int(dataset_input_heightmaps[d][x, z]):320,z] = air_id

        # Real chunks with collapsed caves
        dataset_output_collapsed_caves = np.zeros(shape=(amount, 16, 320, 16), dtype=np.uint8)
        for d in range(amount):
            heightmap_mask = np.arange(320)[None, :, None] < dataset_input_heightmaps[d][:, None, :]
            
            air_mask = dataset_output[d] == air_id
            cave_air_mask = dataset_output[d] == cave_air_id
            collapse_mask = heightmap_mask & (air_mask | cave_air_mask)
            
            dataset_output_collapsed_caves[d] = dataset_output[d]
            dataset_output_collapsed_caves[d][collapse_mask] = stone_id
        
        return dataset_input_heightmaps, dataset_input_caves, dataset_input_volume, dataset_output, dataset_output_collapsed_caves


    def create_fake_chunk():
        fake_chunk = np.zeros((self.Cx, self.Cy, self.Cz), dtype=np.uint8)
        block_ids = [0, 1, 2, 3, 4, 5]
        for x in range(self.Cx):
            for z in range(self.Cz):
                for y in range(0, self.Cy):
                    fake_chunk[x,y,z] = block_ids[random.randint(0, 5)]
        return fake_chunk

    def create_fake_chunk_v2():
        fake_chunk = np.zeros((self.Cx, self.Cy, self.Cz), dtype=np.uint8)
        block_ids = [0, 1, 2, 3, 4, 5]
        for x in range(self.Cx):
            for z in range(self.Cz):
                for y in range(0, 10):
                    fake_chunk[x,y,z] = block_ids[5]
                for y in range(10,50):
                    fake_chunk[x,y,z] = block_ids[3]
                for y in range(50, 70):
                    fake_chunk[x,y,z] = block_ids[4]
                for y in range(70, 100):
                    fake_chunk[x,y,z] = block_ids[3]
                for y in range(100, 140):
                    fake_chunk[x,y,z] = block_ids[random.randint(1, 2)]
                for y in range(140, self.Cy):
                    fake_chunk[x,y,z] = block_ids[0]
        return fake_chunk

    def D_shape(self):
        return False

    def G_shape(self):
        return False



def fast():
    res1 = np.load("res1.npy")
    res2 = data.decode(res1)
    txt = test123.generate_save_file(data.chunks[0,0], res2, "NewModel")
    data.import_save_to_minecraft(txt)               
                

    # ------TESTING------
test123 = None

def test():
    global test123
    amount = 64
    test123 = MCReader(amount, INPUT_HEIGHT, CHUNK_SIZE)
    test123.dump_game_save_file('Single Biome World - Plains')

    test123.create_heightmaps()

    test123.calculate_caves_densities()

    test123.save("DATA123")

def test_load():
    global test123
    amount = 128
    test123 = MCReader(amount, INPUT_HEIGHT, CHUNK_SIZE)
    test123.load("DATA123")

def test_analyze_save_file():
    test123 = MCReader()
    test123.find_highest_block(256, 'Single Biome World - Plains')

#test()
#test_load()

#res1 = np.load("res1.npy")
#res2 = decode(res1)
#test123.chunk_info(0,0)


#test_analyze_save_file()
