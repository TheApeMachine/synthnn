import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

class ResonantNode2D:
    def __init__(self, freq_x=1.0, freq_y=1.0, phase=0.0, amplitude=1.0, node_id=0):
        self.freq_x = freq_x
        self.freq_y = freq_y
        self.phase = phase
        self.amplitude = amplitude
        self.node_id = node_id

    def compute(self, x_grid, y_grid):
        return self.amplitude * np.sin(
            2 * np.pi * (self.freq_x * x_grid + self.freq_y * y_grid) + self.phase
        )

class ResonantNetwork2D:
    def __init__(self, nodes, mode='Ionian'):
        self.nodes = nodes
        self.mode = mode

    def generate_image(self, x_grid, y_grid):
        image = np.zeros_like(x_grid)
        for node in self.nodes:
            image += node.compute(x_grid, y_grid)
        return image

def analyze_foreign_image(image):
    # Compute the 2D FFT
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    # Exclude the DC component and its immediate neighbors
    h, w = image.shape
    center = (h // 2, w // 2)
    magnitude_spectrum[center[0]-1:center[0]+2, center[1]-1:center[1]+2] = 0
    
    # Find the peak frequency
    idx = np.unravel_index(np.argmax(magnitude_spectrum), magnitude_spectrum.shape)
    
    # Calculate the frequencies
    freq_y = idx[0] - center[0]
    freq_x = idx[1] - center[1]
    
    # Normalize frequencies
    freq_y = freq_y / h
    freq_x = freq_x / w
    
    return freq_x, freq_y

def get_mode_intervals(mode):
    modes = {
        'Ionian': [1, 1.25, 1.5, 2, 2.5],
        'Dorian': [1, 1.22, 1.5, 2, 2.33],
        'Lydian': [1, 1.33, 1.67, 2, 2.5],
        'Mixolydian': [1, 1.25, 1.5, 2, 2.33],
        'Aeolian': [1, 1.19, 1.5, 2, 2.25],
    }
    return modes.get(mode, modes['Ionian'])

scaling_factor = 5  # Adjust this value as needed

def retune_nodes(nodes, dominant_freq_x, dominant_freq_y, mode):
    harmonic_ratios = get_mode_intervals(mode)
    for node, ratio in zip(nodes, harmonic_ratios):
        node.freq_x = dominant_freq_x * ratio * scaling_factor
        node.freq_y = dominant_freq_y * ratio * scaling_factor


def main():
    # Step 1: Load and preprocess the foreign image
    foreign_image = Image.open('input_image.jpg').convert('L')  # Convert to grayscale
    foreign_image = foreign_image.resize((256, 256))
    foreign_image_array = np.array(foreign_image)
    
    # Normalize the image
    foreign_image_array = foreign_image_array / 255.0

    # Step 2: Analyze the foreign image
    dominant_freq_x, dominant_freq_y = analyze_foreign_image(foreign_image_array)
    print(f"Dominant Frequencies - X: {dominant_freq_x}, Y: {dominant_freq_y}")

    # Check if dominant frequencies are zero
    if dominant_freq_x == 0 and dominant_freq_y == 0:
        print("Dominant frequencies are zero after excluding DC component.")
        print("This may occur if the image lacks significant frequency components.")
        print("Try using a different image with more texture or patterns.")
        return
    
    # Step 3: Initialize nodes
    num_nodes = 256  # Number of nodes in the network
    nodes = [ResonantNode2D(node_id=i+1) for i in range(num_nodes)]
    
    # Step 4: Retune nodes based on the foreign image and selected mode
    selected_mode = 'Dorian'  # You can change the mode here
    retune_nodes(nodes, dominant_freq_x, dominant_freq_y, selected_mode)
    
    # Step 5: Generate new image
    x = np.linspace(0, 1, foreign_image_array.shape[1])
    y = np.linspace(0, 1, foreign_image_array.shape[0])
    x_grid, y_grid = np.meshgrid(x, y)
    
    network = ResonantNetwork2D(nodes, mode=selected_mode)
    generated_image = network.generate_image(x_grid, y_grid)
    
    # Normalize the generated image
    generated_image = (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min())
    
    # Step 6: Display the images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(foreign_image_array, cmap='gray')
    plt.title('Input (Foreign) Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(generated_image, cmap='gray')
    plt.title(f'Generated Image (Mode: {selected_mode})')
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    main()
