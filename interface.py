from N_mc2data import MCReader, Block
import ChunkGAN
import numpy as np
import math

data = MCReader(64)
data.load("DATA123")

model = "one-hot_no-cave-loss_full-dis-unet"

ChunkGAN.build()
ChunkGAN.load_model(model)

world_xz = 6
a_half = world_xz
amount = a_half * a_half

HMAP = np.zeros((a_half*16,a_half*16), dtype=np.uint8)
for x in range(a_half*16):
    for z in range(a_half*16):
        HMAP[x, z] = int(4*math.sin(z/8) + (8*math.sin(x/8) + 126))


HMAP_split = np.zeros((a_half, a_half, 16, 16), dtype=np.uint8)
for x in range(a_half):
    for z in range(a_half):
        HMAP_split[x, z] = HMAP[(x*16):(x*16)+16, (z*16):(z*16)+16]

DENS = 2500

heightmaps = np.zeros((amount, 16, 16), dtype=np.uint8)
for QQ in range(0, amount):
    Wx = QQ % a_half
    Wz = QQ // a_half
    heightmaps[QQ] = data.heightmaps[45 + Wx, 35 + Wz]
    #heightmaps[QQ] = HMAP_split[Wx, Wz]

# 45, 35
    
#cave_dens = np.reshape(data.cave_densities[0,0], (1, 1))
cave_dens = np.reshape([DENS] * amount, (amount, 1))
seed = 45

chunks, cave_maps = ChunkGAN.generate(amount, heightmaps, cave_dens, seed)

# decode
world_width = int(math.sqrt(np.shape(chunks)[0]))

world = np.zeros((world_width, world_width, 16, 320, 16), dtype=np.uint8)

for x in range(np.shape(world)[0]):
    for z in range(np.shape(world)[1]):
        world[x,z] = data.decode(chunks[(z*world_width) + x])

save = True
if save == True:                
    path = data.generate_save_file_2(world, "Test_World")
    data.import_save_to_minecraft(path) 


real_blocks = {
    'air': [],
    'dirt': [],
    'sand': [],
    'stone': [],
    'cave_air': [],
    'bedrock': []
    }

generated_blocks = {
    'air': [],
    'dirt': [],
    'sand': [],
    'stone': [],
    'cave_air': [],
    'bedrock': []
    }

generated_blocks['air'].append(np.count_nonzero(world == 0))
generated_blocks['dirt'].append(np.count_nonzero(world == 1))
generated_blocks['sand'].append(np.count_nonzero(world == 2))
generated_blocks['stone'].append(np.count_nonzero(world == 3))
generated_blocks['cave_air'].append(np.count_nonzero(world == 4))
generated_blocks['bedrock'].append(np.count_nonzero(world == 5))

_, _, _, real, _ = data.get_ml_input_set(0, amount)

real_blocks['air'].append(np.count_nonzero(real == 0))
real_blocks['dirt'].append(np.count_nonzero(real == 1))
real_blocks['sand'].append(np.count_nonzero(real == 2))
real_blocks['stone'].append(np.count_nonzero(real == 3))
real_blocks['cave_air'].append(np.count_nonzero(real == 4))
real_blocks['bedrock'].append(np.count_nonzero(real == 5))

print(real_blocks)
print(generated_blocks)
