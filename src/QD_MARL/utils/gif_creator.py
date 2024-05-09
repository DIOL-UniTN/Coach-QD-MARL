import argparse
import os
from PIL import Image

def gif_creator(path):
    filenames = [f for f in os.listdir(path) if f.endswith('.png')]
    filenames = sorted(filenames, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    images = [Image.open(os.path.join(path, f)) for f in filenames]
    frames = images[0]
    output_file = os.path.join(path, 'output.gif')
    frames.save(output_file, format='GIF', append_images=images, save_all=True, duration=400, loop=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("imgs_folder", help="folder with imgs to create gif")
    path = parser.parse_args().imgs_folder
    gif_creator(path)

