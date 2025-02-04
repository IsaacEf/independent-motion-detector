# independent-motion-detector
The goal of this project is to detect independently moving objects in a scene captured by a single camera. This involves estimating sparse motion vectors, determining whether the camera itself is moving by locating its focus of expansion (foe), and identifying motion that deviates from the camera’s movement.

# How to run it
To run this program, run the command below ensuring to use the correct two images, replacing "image1" and "image2" with the appropirate file names.

'''console
draw_all("data/image1.png", "data/image2.png")
'''
