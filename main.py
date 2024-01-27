import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import soundfile as sf
import sounddevice as sd

class ImageSeparationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image and Speech Separation App")

        self.input_type_var = tk.StringVar()
        self.input_type_var.set("Image")

        self.method_var = tk.StringVar()
        self.method_var.set("Kurtosis")

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Select Input Type:").pack()
        tk.Radiobutton(self.root, text="Image", variable=self.input_type_var, value="Image").pack()
        tk.Radiobutton(self.root, text="Speech", variable=self.input_type_var, value="Speech").pack()

        tk.Label(self.root, text="Select Separation Method:").pack()
        tk.OptionMenu(self.root, self.method_var, "Kurtosis", "Gram Schmidt", "Negentropy").pack()

        tk.Button(self.root, text="Choose File", command=self.choose_file).pack()

    def choose_file(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        if self.input_type_var.get() == "Image":
            self.perform_image_separation(file_path)
        elif self.input_type_var.get() == "Speech":
            self.perform_audio_separation(file_path, self.method_var.get())

    def perform_image_separation(self, image_path):
        S1 = plt.imread(image_path).astype(float)
        S2 = plt.imread(image_path).astype(float)

        A = np.array([[0.8, 0.2], [1/2, 2/3]])
        X1 = A[0, 0] * S1 + A[0, 1] * S2
        X2 = A[1, 0] * S1 + A[1, 1] * S2

        m, n, _ = X1.shape
        x1 = X1.flatten() - np.mean(X1)
        x2 = X2.flatten() - np.mean(X2)

        if self.method_var.get() == "Kurtosis":
            W = np.random.rand(2, 2)

            max_iterations = 1000
            learning_rate = 0.01
            threshold = 1e-6

            prev_negentropy1 = 0
            prev_negentropy2 = 0

            for iteration in range(1, max_iterations + 1):
                X1_bar = np.dot(W[0, :], [x1, x2])
                X2_bar = np.dot(W[1, :], [x1, x2])

                negentropy1 = kurtosis(X1_bar)
                negentropy2 = kurtosis(X2_bar)

                gradient1 = 4 * np.mean(X1_bar**3 * X1_bar) - 3 * np.mean(X1_bar**2)
                gradient2 = 4 * np.mean(X2_bar**3 * X2_bar) - 3 * np.mean(X2_bar**2)

                W = W + learning_rate * np.outer([gradient1, gradient2], [x1, x2])

                if (abs(negentropy1 - prev_negentropy1) < threshold) and \
                        (abs(negentropy2 - prev_negentropy2) < threshold):
                    break
                prev_negentropy1 = negentropy1
                prev_negentropy2 = negentropy2

            U1 = np.reshape(X1_bar, (m, n))
            U2 = np.reshape(X2_bar, (m, n))

        elif self.method_var.get() == "Gram Schmidt":
            v1 = x1 - np.mean(x1)
            v2 = x2 - np.mean(x2)
            u1 = v1 / np.linalg.norm(v1)
            proj_v2_u1 = np.dot(u1.T, v2)
            u2 = v2 - proj_v2_u1 * u1
            u2 = u2 / np.linalg.norm(u2)

            U1 = u1.reshape((m, n, -1))
            U2 = u2.reshape((m, n, -1))

            # Normalize components differently
            U1 = (U1 - np.min(U1)) / (np.max(U1) - np.min(U1)) * 255
            U2 = (U2 - np.min(U2)) / (np.max(U2) - np.min(U2)) * 255

        elif self.method_var.get() == "Negentropy":
            W = np.random.rand(2, 2)

            max_iterations = 1000
            learning_rate = 0.01
            threshold = 1e-6

            prev_negentropy1 = 0
            prev_negentropy2 = 0

            for iteration in range(1, max_iterations + 1):
                X1_bar = np.dot(W[0, :], [x1, x2])
                X2_bar = np.dot(W[1, :], [x1, x2])

                negentropy1 = kurtosis(X1_bar)
                negentropy2 = kurtosis(X2_bar)

                gradient1 = 4 * np.mean(X1_bar ** 3 * X1_bar) - 3 * np.mean(X1_bar ** 2)
                gradient2 = 4 * np.mean(X2_bar ** 3 * X2_bar) - 3 * np.mean(X2_bar ** 2)

                # Use element-wise multiplication to update W
                W = W + learning_rate * np.array([[gradient1, gradient2], [gradient1, gradient2]]) * np.vstack([x1, x2])

                if (abs(negentropy1 - prev_negentropy1) < threshold) and \
                        (abs(negentropy2 - prev_negentropy2) < threshold):
                    break
                prev_negentropy1 = negentropy1
                prev_negentropy2 = negentropy2

            U1 = np.reshape(X1_bar, (m, n))
            U2 = np.reshape(X2_bar, (m, n))

        # Display independent components
        plt.figure()
        plt.subplot(1, 2, 1), plt.imshow(U1.astype(np.uint8)), plt.title('Separated Component 1')
        plt.subplot(1, 2, 2), plt.imshow(U2.astype(np.uint8)), plt.title('Separated Component 2')
        plt.show()

    def perform_audio_separation(self, audio_path, method):
        data, fs = sf.read(audio_path, dtype='float32')

        if method == "Kurtosis":
            W = np.random.rand(2, 2)

            max_iterations = 1000
            learning_rate = 0.01
            threshold = 1e-6

            prev_negentropy1 = 0
            prev_negentropy2 = 0

            for iteration in range(1, max_iterations + 1):
                X1_bar = np.dot(W[0, :], [data, data])
                X2_bar = np.dot(W[1, :], [data, data])

                negentropy1 = kurtosis(X1_bar)
                negentropy2 = kurtosis(X2_bar)

                gradient1 = 4 * np.mean(X1_bar**3 * X1_bar) - 3 * np.mean(X1_bar**2)
                gradient2 = 4 * np.mean(X2_bar**3 * X2_bar) - 3 * np.mean(X2_bar**2)

                # Use element-wise multiplication to update W
                W = W + learning_rate * np.array([[gradient1, gradient2], [gradient1, gradient2]]) * np.vstack([data, data])

                if (abs(negentropy1 - prev_negentropy1) < threshold) and \
                        (abs(negentropy2 - prev_negentropy2) < threshold):
                    break
                prev_negentropy1 = negentropy1
                prev_negentropy2 = negentropy2

            output1 = np.reshape(X1_bar, (len(data),))
            output2 = np.reshape(X2_bar, (len(data),))

        elif method == "Gram Schmidt":
            v1 = data - np.mean(data)
            v2 = np.random.rand(len(data))
            v2 = v2 - np.dot(v1, v2) / np.dot(v1, v1) * v1
            u1 = v1 / np.linalg.norm(v1)
            u2 = v2 / np.linalg.norm(v2)

            U1 = u1.reshape((len(data), -1))
            U2 = u2.reshape((len(data), -1))

            # Normalize components differently
            U1 = (U1 - np.min(U1)) / (np.max(U1) - np.min(U1))
            U2 = (U2 - np.min(U2)) / (np.max(U2) - np.min(U2))

            U1 = U1 / np.max(np.abs(U1))  # Normalize to ensure values are in the range [-1, 1]
            U2 = U2 / np.max(np.abs(U2))

            output1 = np.dot(data, U1)
            output2 = np.dot(data, U2)

        elif method == "Negentropy":
            W = np.random.rand(2, 2)

            max_iterations = 1000
            learning_rate = 0.01
            threshold = 1e-6

            prev_negentropy1 = 0
            prev_negentropy2 = 0

            for iteration in range(1, max_iterations + 1):
                X1_bar = W[0, 0] * data + W[0, 1] * data
                X2_bar = W[1, 0] * data + W[1, 1] * data

                negentropy1 = kurtosis(X1_bar)
                negentropy2 = kurtosis(X2_bar)

                gradient1 = 4 * np.mean((X1_bar**3) * X1_bar) - 3 * np.mean(X1_bar**2)
                gradient2 = 4 * np.mean((X2_bar**3) * X2_bar) - 3 * np.mean(X2_bar**2)

                # Use element-wise multiplication to update W
                W = W + learning_rate * np.array([[gradient1, gradient2], [gradient1, gradient2]]) * np.vstack([data, data])

                if (abs(negentropy1 - prev_negentropy1) < threshold) and \
                        (abs(negentropy2 - prev_negentropy2) < threshold):
                    break
                prev_negentropy1 = negentropy1
                prev_negentropy2 = negentropy2

            output1 = np.reshape(X1_bar, (len(data),))
            output2 = np.reshape(X2_bar, (len(data),))

        # Normalize and play the separated audio signals
        output1_normalized = output1 / np.max(np.abs(output1))
        output2_normalized = output2 / np.max(np.abs(output2))

        sd.play(output1_normalized, fs)
        sd.wait()
        sd.play(output2_normalized, fs)
        sd.wait()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSeparationApp(root)
    root.mainloop()
