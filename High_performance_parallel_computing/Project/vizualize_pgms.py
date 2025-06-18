import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio.v2 as imageio
import glob

def update(frame_idx):
    print(f"Displaying frame {frame_idx + 1}/{len(frames)}")
    im.set_array(frames[frame_idx])
    return [im]

###Loading the frames###
pgm_files = sorted(glob.glob('frames/snapshot_*.pgm'), key=lambda f: float(f.split('_')[-1].replace('.pgm', '')))
frames = [imageio.imread(f) for f in pgm_files]
print(pgm_files)

####Creating the figure###
fig, ax = plt.subplots()
im = ax.imshow(frames[0], cmap='gray', animated=True)
ax.axis('off')

###Animation object###
ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=300, blit=False, repeat=True)

plt.show()
