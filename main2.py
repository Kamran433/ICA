from flask import Flask, render_template, request, send_file,send_from_directory
from flask_socketio import SocketIO
import base64
import os
import numpy as np
import librosa
from scipy.io import wavfile
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis, entropy
from skimage import io, color
from skimage.util import img_as_ubyte
from scipy.signal import resample
from PIL import Image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def emit_status(message):
    socketio.emit('status', message)

def preprocess_audio(signal, sr):
    emit_status('Preprocessing audio...')
    target_sr = 44100
    if sr != target_sr:
        signal = resample(signal, int(len(signal) * target_sr / sr))
    return signal

def preprocess_image(image_path):
    emit_status('Preprocessing image...')
    img = io.imread(image_path)

    # Additional preprocessing steps for various image formats
    if img.ndim == 2:
        img = color.gray2rgb(img)

    img = img_as_ubyte(img)
    img = normalize_image(img)
    return img

def normalize_image(img):
    emit_status('Normalizing image...')
    img_min, img_max = np.min(img), np.max(img)
    img = (img - img_min) / (img_max - img_min)
    img = 2 * img - 1
    return img

def convert_audio(file_path):
    emit_status('Converting audio...')
    # Additional steps to convert audio to a common format if needed
    # For example, you might want to convert MP3 to WAV
    return file_path  # Placeholder, modify based on your requirements

def convert_image(file_path):
    emit_status('Converting image...')
    # Additional steps to convert image to a common format if needed
    # For example, you might want to convert different image formats to PNG
    img = Image.open(file_path)
    converted_path = os.path.join(app.config['UPLOAD_FOLDER'], 'converted_image.png')
    img.save(converted_path)
    return converted_path

def separate_audio_kurtosis(mixed_signal):
    emit_status('Separating audio using Kurtosis...')
    kurt = kurtosis(mixed_signal)
    separated_signals = mixed_signal / kurt
    return separated_signals

def separate_audio_negentropy(mixed_signal):
    emit_status('Separating audio using Negentropy...')
    negentropies = np.apply_along_axis(negentropy, axis=1, arr=mixed_signal)
    separated_signals = mixed_signal / negentropies[:, np.newaxis]
    return separated_signals

def gram_schmidt(matrix):
    emit_status('Applying Gram-Schmidt to audio...')
    Q, R = np.linalg.qr(matrix)
    return Q

def separate_audio_gram_schmidt(mixed_signal):
    return gram_schmidt(mixed_signal)

def separate_audio_ica(mixed_signal, method='fastica'):
    if method == 'fastica':
        emit_status('Separating audio using FastICA...')
        ica = FastICA(n_components=1)
        separated_signal = ica.fit_transform(mixed_signal.reshape(-1, 1)).flatten()
    elif method == 'kurtosis':
        separated_signal = separate_audio_kurtosis(mixed_signal)
    elif method == 'negentropy':
        separated_signal = separate_audio_negentropy(mixed_signal)
    elif method == 'gram_schmidt':
        separated_signal = separate_audio_gram_schmidt(mixed_signal)
    else:
        ica = FastICA(n_components=1)
        separated_signal = ica.fit_transform(mixed_signal.reshape(-1, 1)).flatten()

    return separated_signal

def process_audio(file_path, ica_method='fastica'):
    separated_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'separated_audio')
    os.makedirs(separated_folder, exist_ok=True)

    emit_status('Processing audio...')
    sr, mixed_audio = wavfile.read(file_path)
    mixed_audio = preprocess_audio(mixed_audio, sr)

    # Convert audio if needed
    converted_audio_path = convert_audio(file_path)
    if converted_audio_path:
        emit_status(f'Using converted audio: {converted_audio_path}')
        sr, mixed_audio = wavfile.read(converted_audio_path)

    mixed_audio = mixed_audio.reshape(-1, 1)
    separated_audio = separate_audio_ica(mixed_audio, method=ica_method)
    separated_audio_path = os.path.join(separated_folder, 'separated_audio.wav')

    # Save the separated audio
    wavfile.write(separated_audio_path, sr, separated_audio)

    return sr, separated_audio

def process_image(file_path, ica_methods=['fastica', 'kurtosis', 'negentropy', 'gram_schmidt']):
    img = preprocess_image(file_path)
    separated_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'separated_image')
    os.makedirs(separated_folder, exist_ok=True)
    channel_paths = separate_image(img, ica_methods)

    # Convert image if needed
    converted_image_path = convert_image(file_path)
    if converted_image_path:
        emit_status(f'Using converted image: {converted_image_path}')
        img = preprocess_image(converted_image_path)

    # Convert images to base64 encoding
    base64_channels = []
    for i, channel in enumerate(channel_paths):
        channel_filepath = os.path.join(separated_folder, f'separated_channel_{i + 1}.png')
        io.imsave(channel_filepath, channel)
        with open(channel_filepath, "rb") as image_file:
            base64_channel = base64.b64encode(image_file.read()).decode('utf-8')
            base64_channels.append(base64_channel)

    return base64_channels

def separate_image_kurtosis(img):
    emit_status('Separating image using Kurtosis...')
    separated_channel1 = img[:, :, 0] / (kurtosis(img[:, :, 0].ravel()) + 1e-10)
    separated_channel2 = img[:, :, 1] / (kurtosis(img[:, :, 1].ravel()) + 1e-10)
    separated_channel3 = img[:, :, 2] / (kurtosis(img[:, :, 2].ravel()) + 1e-10)
    return np.stack([separated_channel1, separated_channel2, separated_channel3], axis=-1)

def separate_image_negentropy(img):
    emit_status('Separating image using Negentropy...')
    if img.shape[-1] == 4:
        # Convert RGBA to RGB by removing the alpha channel
        img = img[:, :, :3]

    gray_img = color.rgb2gray(img)
    separated_channel1 = gray_img / negentropy(gray_img.ravel())
    separated_channel1 = np.clip(separated_channel1, 0, 1)
    separated_channel1_uint8 = img_as_ubyte(separated_channel1)
    return separated_channel1_uint8

def separate_image_gram_schmidt(img):
    emit_status('Applying Gram-Schmidt to image...')
    separated_channel1 = gram_schmidt(img[:, :, 0])
    separated_channel2 = gram_schmidt(img[:, :, 1])
    separated_channel3 = gram_schmidt(img[:, :, 2])
    return np.stack([separated_channel1, separated_channel2, separated_channel3], axis=-1)

def separate_image(img, ica_methods=['fastica', 'kurtosis', 'negentropy', 'gram_schmidt']):
    separated_channels = []

    for ica_method in ica_methods:
        if ica_method == 'fastica':
            ica = FastICA(n_components=3)
            separated_channel = ica.fit_transform(img.reshape(-1, 3)).reshape(img.shape)
        elif ica_method == 'kurtosis':
            separated_channel = separate_image_kurtosis(img)
        elif ica_method == 'negentropy':
            separated_channel = separate_image_negentropy(img)
        elif ica_method == 'gram_schmidt':
            separated_channel = separate_image_gram_schmidt(img)
        else:
            ica = FastICA(n_components=3)
            separated_channel = ica.fit_transform(img.reshape(-1, 3)).reshape(img.shape)

        separated_channel = (separated_channel - np.nanmin(separated_channel)) / (np.nanmax(separated_channel) - np.nanmin(separated_channel) + 1e-10)
        separated_channel_uint8 = img_as_ubyte(separated_channel)

        if len(separated_channel_uint8.shape) == 2:
            separated_channel_uint8 = np.stack([separated_channel_uint8] * 3, axis=-1)

        separated_channels.append(separated_channel_uint8)

    return separated_channels

def negentropy(x):
    emit_status('Calculating negentropy...')
    if isinstance(x, (int, float)):
        return entropy(np.exp(-(x**2) / 2)) - entropy(np.random.normal(size=1))
    elif isinstance(x, np.ndarray):
        return entropy(np.exp(-(x**2) / 2)) - entropy(np.random.normal(size=x.shape))
    else:
        raise ValueError("Input must be either scalar or NumPy array.")

@app.route('/play_audio')
def play_audio():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'separated_audio', 'separated_audio.wav')
    return send_file(file_path)


@app.route('/download_audio')
def download_audio():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'separated_audio', 'separated_audio.wav')
    return send_file(file_path, as_attachment=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        separation_type = request.form['separation_type']
        ica_method = request.form['ica_method']

        if separation_type == 'audio':
            file = request.files['file']
            if file.filename != '':
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_audio.wav')
                file.save(file_path)
                sr, separated_audio = process_audio(file_path, ica_method=ica_method)
                separated_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'separated_audio', 'separated_audio.wav')
                # Save the separated audio
                wavfile.write(separated_audio_path, sr, separated_audio)
                emit_status('Audio separation complete!')
                return render_template('result.html', sr=sr, audio=separated_audio, audio_path=separated_audio_path)

        elif separation_type == 'image':
            file = request.files['file']
            if file.filename != '':
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.png')
                file.save(file_path)
                channel_paths = process_image(file_path, ica_methods=['fastica', 'kurtosis', 'negentropy', 'gram_schmidt'])
                emit_status('Image separation complete!')
                return render_template('image_result.html', channel_paths=channel_paths)

    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
