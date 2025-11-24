import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data

def change_pitch(data, sr, n_steps=2):
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)

def change_speed(data, rate=1.2):
    return librosa.effects.time_stretch(data, rate=rate)

# Load audio
file_path = 'common_voice_en_42696167.mp3'
y, sr = librosa.load(file_path, duration=5)

# Apply augmentations
y_noise = add_noise(y)
y_pitch = change_pitch(y, sr)
y_speed = change_speed(y)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Vizuelizacija Augmentacije Podataka', fontsize=16)

# Original
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[0, 0])
axes[0, 0].set_title('Originalni Signal')

# Noise
D_noise = librosa.amplitude_to_db(np.abs(librosa.stft(y_noise)), ref=np.max)
librosa.display.specshow(D_noise, sr=sr, x_axis='time', y_axis='log', ax=axes[0, 1])
axes[0, 1].set_title('Dodat Šum (White Noise)')

# Pitch
D_pitch = librosa.amplitude_to_db(np.abs(librosa.stft(y_pitch)), ref=np.max)
librosa.display.specshow(D_pitch, sr=sr, x_axis='time', y_axis='log', ax=axes[1, 0])
axes[1, 0].set_title('Promena Visine Tona (Pitch Shift)')

# Speed
D_speed = librosa.amplitude_to_db(np.abs(librosa.stft(y_speed)), ref=np.max)
librosa.display.specshow(D_speed, sr=sr, x_axis='time', y_axis='log', ax=axes[1, 1])
axes[1, 1].set_title('Promena Brzine (Time Stretch)')

plt.tight_layout()
plt.savefig('docs/images/augmentation_visualization.png')
print("Image saved to docs/images/augmentation_visualization.png")
