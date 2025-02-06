# independent-motion-detector
The goal of this project is to detect independently moving objects in a scene captured by a single camera. This involves estimating sparse motion vectors, determining whether the camera itself is moving by locating its focus of expansion (foe), and identifying motion that deviates from the camera’s movement.

# How to run it
To run this program, run the command below ensuring to use the correct two images, replacing "image1" and "image2" with the appropriate file names.

```python 
draw_all("data/image1.png", "data/image2.png")
```
# Example Output


<img width="1049" alt="Screenshot 2025-02-05 at 3 52 30 PM" src="https://github.com/user-attachments/assets/ce3e5987-ff01-45c8-9afd-3f002e07167d" />
<img width="1066" alt="Screenshot 2025-02-05 at 3 52 39 PM" src="https://github.com/user-attachments/assets/d5d416f4-5963-49ec-b69f-756ef8f39668" />
