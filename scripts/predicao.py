import torch
import numpy as np
import librosa
import soundfile as sf

def predict_species(model, audio_path, classes,
                    sr=22050,
                    segment_duration=3.0,
                    segment_hop=1.5,
                    n_fft=2048,
                    hop_length=512,
                    n_mels=128,
                    device="cpu"):

    # ---- 1. carregar áudio (wav/mp3/mp4/etc) ----
    try:
        y, orig_sr = sf.read(audio_path)
        if len(y.shape) == 2:
            y = y.mean(axis=1)
        if orig_sr != sr:
            y = librosa.resample(y.astype(float), orig_sr=orig_sr, target_sr=sr)
    except Exception:
        try:
            y, _ = librosa.load(audio_path, sr=sr, mono=True)
        except Exception as e:
            raise RuntimeError(f"Falha ao carregar áudio: {e}")

    # ---- 2. segmentar áudio ----
    seg_samples = int(segment_duration * sr)
    hop_samples = int(segment_hop * sr)

    segments = []
    if len(y) < seg_samples:
        pad_len = seg_samples - len(y)
        y_seg = np.pad(y, (0, pad_len))
        segments.append(y_seg)
    else:
        for start in range(0, len(y) - seg_samples + 1, hop_samples):
            segments.append(y[start:start+seg_samples])

    # ---- 3. gerar espectrogramas ----
    specs = []
    for seg in segments:
        mel = librosa.feature.melspectrogram(
            y=seg,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # normalização local
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        specs.append(mel_db.astype(np.float32))

    # ---- 4. pad temporal ----
    tensors = [torch.tensor(x).unsqueeze(0) for x in specs]  # (1, 128, T)
    max_len = max(t.size(-1) for t in tensors)

    padded = []
    for t in tensors:
        if t.size(-1) < max_len:
            t = torch.nn.functional.pad(t, (0, max_len - t.size(-1)))
        padded.append(t)

    batch = torch.stack(padded)
    batch = batch.to(next(model.parameters()).device)

    # ---- 5. inferência ----
    model.eval()
    with torch.no_grad():
        logits = model(batch)            # (B, num_classes)
        probs = torch.softmax(logits, -1)

    # ---- 6. agregação temporal ----
    mean_prob = probs.mean(0)           # (num_classes,)

    # ---- 7. top-1 ----
    pred_idx = mean_prob.argmax().item()
    pred_class = classes[pred_idx]
    pred_conf = mean_prob[pred_idx].item()

    # ---- 8. top-5 ----
    top5_vals, top5_idx = torch.topk(mean_prob, 5)
    top5_idx = top5_idx.cpu().numpy()
    top5_vals = top5_vals.cpu().numpy()

    top5 = [(classes[i], float(top5_vals[k])) for k, i in enumerate(top5_idx)]

    return {
        "top1": (pred_class, pred_conf),
        "top5": top5,
        "probs": mean_prob.cpu().numpy()
    }
