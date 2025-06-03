import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Tool")

        # Variables
        self.image_path = None
        self.original_image = None
        self.processed_image = None

        # GUI Elements
        self.btn_upload = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.btn_upload.pack(pady=10)

        self.btn_grayscale = tk.Button(root, text="RGB → Grayscale", command=self.rgb_to_grayscale)
        self.btn_grayscale.pack(pady=5)

        self.btn_rgb_to_hsv = tk.Button(root, text="RGB → HSV", command=self.rgb_to_hsv)
        self.btn_rgb_to_hsv.pack(pady=5)

        self.btn_laplacian = tk.Button(root, text="Laplacian Edge Detection", command=self.laplacian_edge)
        self.btn_laplacian.pack(pady=5)

        self.btn_butterworth = tk.Button(root, text="Butterworth High-Pass", command=self.butterworth_highpass)
        self.btn_butterworth.pack(pady=5)

        self.btn_save = tk.Button(root, text="Save Processed Image", command=self.save_image)
        self.btn_save.pack(pady=5)

        self.frame_original = tk.Frame(root)
        self.frame_original.pack(side=tk.LEFT, padx=10, pady=10)

        self.frame_processed = tk.Frame(root)
        self.frame_processed.pack(side=tk.RIGHT, padx=10, pady=10)

        self.label_original_text = tk.Label(self.frame_original, text="Original Image")
        self.label_original_text.pack()

        self.label_original_img = tk.Label(self.frame_original)
        self.label_original_img.pack()

        self.label_processed_text = tk.Label(self.frame_processed, text="Processed Image")
        self.label_processed_text.pack()

        self.label_processed_img = tk.Label(self.frame_processed)
        self.label_processed_img.pack()

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            self.original_image = Image.open(self.image_path).convert("RGB")
            self.display_image(self.original_image, self.label_original_img)

    def display_image(self, image, label):
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

    def rgb_to_grayscale(self):
        if self.original_image:
            img_array = np.array(self.original_image)
            gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            self.processed_image = Image.fromarray(gray)
            self.display_image(self.processed_image, self.label_processed_img)

    def rgb_to_hsv(self):
        if self.original_image:
            img_array = np.array(self.original_image.convert("RGB")) / 255.0
            hsv_img = rgb_to_hsv(img_array)
            rgb_converted = (hsv_to_rgb(hsv_img) * 255).astype(np.uint8)
            self.processed_image = Image.fromarray(rgb_converted)
            self.display_image(self.processed_image, self.label_processed_img)

    def laplacian_edge(self):
        if self.original_image:
            gray = np.array(self.original_image.convert('L'))
            
            # Improved Laplacian kernel (normalized)
            kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]]) / 4
            
            # Pad the image to handle borders
            padded = np.pad(gray, 1, mode='constant')
            output = np.zeros_like(gray)
            
            # Apply convolution
            for i in range(gray.shape[0]):
                for j in range(gray.shape[1]):
                    region = padded[i:i+3, j:j+3]
                    output[i, j] = np.sum(region * kernel)
            
            # Take absolute value and clip to 0-255
            edges = np.clip(np.abs(output), 0, 255).astype(np.uint8)
            
            self.processed_image = Image.fromarray(edges)
            self.display_image(self.processed_image, self.label_processed_img)

    def butterworth_highpass(self, cutoff=30, order=2):
        if self.original_image:
            gray = np.array(self.original_image.convert('L'))
            fft_img = fftshift(fft2(gray))
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2

            # Butterworth High-Pass Filter
            mask = np.zeros((rows, cols))
            for i in range(rows):
                for j in range(cols):
                    d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2 + 1e-5) 
                    mask[i, j] = 1 / (1 + (cutoff / d) ** (2 * order))

            filtered_fft = fft_img * mask
            filtered_img = np.abs(ifft2(ifftshift(filtered_fft)))
            filtered_img = (filtered_img / filtered_img.max() * 255).astype(np.uint8)

            self.processed_image = Image.fromarray(filtered_img)
            self.display_image(self.processed_image, self.label_processed_img)

    def save_image(self):
        if self.processed_image:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")],
                title="Save Processed Image As"
            )
            if save_path:
                self.processed_image.save(save_path)
                messagebox.showinfo("Success", "Image saved successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
