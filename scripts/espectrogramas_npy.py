import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm

# =========================
# CONFIGURAÇÕES
# =========================
DATASET_DIR = "dataset"
LABELS_CSV = "labels.csv"
OUTPUT_SPECS = "spectrograms_new_npy"

os.makedirs(OUTPUT_SPECS, exist_ok=True)

SR = 22050
N_FFT = 2048
HOP = 512
N_MELS = 128

# "png" ou "npy" ou "both"
SAVE_MODE = "npy"

# janelas de áudio (em segundos)
SEGMENT_DURATION = 3.0   # duração de cada janela
SEGMENT_HOP = 1.5        # passo entre janelas (overlap de 50%)


# =========================
# Função robusta para ler áudio
# =========================
def load_audio_safe(path, target_sr=SR):
    try:
        y, sr = sf.read(path, always_2d=False)

        if y is None or y.size == 0:
            raise ValueError("Áudio vazio ou corrompido")

        # estéreo -> mono
        if len(y.shape) == 2:
            y = y.mean(axis=1)

        # substitui NaN
        if np.isnan(y).any():
            y = np.nan_to_num(y)

        # resample
        if sr != target_sr:
            y = librosa.resample(y.astype(float), orig_sr=sr, target_sr=target_sr)

        return y, target_sr

    except Exception:
        return None, None


# =========================
# 1. Carregar labels
# =========================
df = pd.read_csv(LABELS_CSV)
species_list = df["species"].unique()

print(f"Total de espécies: {len(species_list)}")


# =========================
# 2. Gerar espectrogramas em janelas
# =========================
segment_samples = int(SEGMENT_DURATION * SR)
hop_samples = int(SEGMENT_HOP * SR)

for species in species_list:

    print(f"\nProcessando espécie: {species}")

    species_dir = os.path.join(OUTPUT_SPECS, species)
    os.makedirs(species_dir, exist_ok=True)

    df_species = df[df["species"] == species].reset_index(drop=True)

    file_index = 1

    for _, row in tqdm(df_species.iterrows(), total=len(df_species)):
        audio_path = os.path.join(DATASET_DIR, row["path"])

        y, sr = load_audio_safe(audio_path)

        if y is None:
            print(f"[IGNORADO] {audio_path} → arquivo corrompido")
            continue

        # -----------------------------
        # criar janelas do áudio
        # -----------------------------
        if len(y) < segment_samples:
            # se muito curto, pad até o tamanho da janela
            pad_len = segment_samples - len(y)
            y_seg = np.pad(y, (0, pad_len))
            segments = [y_seg]
        else:
            segments = []
            for start in range(0, len(y) - segment_samples + 1, hop_samples):
                end = start + segment_samples
                segments.append(y[start:end])

        # -----------------------------
        # gerar espectrograma para cada janela
        # -----------------------------
        for seg in segments:
            try:
                mel = librosa.feature.melspectrogram(
                    y=seg,
                    sr=SR,
                    n_fft=N_FFT,
                    hop_length=HOP,
                    n_mels=N_MELS
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)

                # -------- salvar como .npy --------
                if SAVE_MODE in ("npy", "both"):
                    npy_path = os.path.join(species_dir, f"{file_index}.npy")
                    np.save(npy_path, mel_db.astype(np.float32))

                # -------- salvar como PNG --------
                if SAVE_MODE in ("png", "both"):
                    png_path = os.path.join(species_dir, f"{file_index}.png")
                    plt.figure(figsize=(3, 3))
                    librosa.display.specshow(mel_db, sr=SR, hop_length=HOP, cmap="magma")
                    plt.axis("off")
                    plt.tight_layout(pad=0)
                    plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
                    plt.close()

                file_index += 1

            except Exception as e:
                print(f"[ERRO] Falha ao processar {audio_path} (janela): {e}")
                continue

print("\nFinalizado!")
print(f"Espectrogramas salvos em: {OUTPUT_SPECS}")
