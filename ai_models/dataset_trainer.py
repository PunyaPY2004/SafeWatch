"""
SafeWatch Dataset Training & Integration
==========================================
This file handles:
1. Violence Detection Training (RWF-2000 dataset)
2. Audio Classification Training (ESC-50 dataset)
3. Night Mode Testing (ExDark dataset)
4. Loitering/Tracking Setup (MOT17 dataset)
5. Testing all trained models

HOW TO USE:
-----------
Step 1: Download datasets and put in datasets/ folder
Step 2: Run: python dataset_trainer.py --task all
Step 3: Trained models saved to trained_models/ folder
Step 4: Models auto-loaded by ai_engine.py

DATASET FOLDER STRUCTURE:
--------------------------
datasets/
├── rwf2000/
│   ├── train/
│   │   ├── Fight/        ← fight videos (.avi/.mp4)
│   │   └── NonFight/     ← normal videos
│   └── val/
│       ├── Fight/
│       └── NonFight/
├── esc50/
│   ├── audio/            ← .wav files
│   └── meta/
│       └── esc50.csv     ← labels file
├── mot17/
│   └── train/
│       └── MOT17-01/     ← sequence folders
└── exdark/
    └── images/           ← dark images
"""

import os
import sys
import cv2
import numpy as np
import json
import argparse
from datetime import datetime

# ===== PATHS =====
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR   = os.path.join(BASE_DIR, "datasets")
MODELS_DIR     = os.path.join(BASE_DIR, "trained_models")
REPORTS_DIR    = os.path.join(BASE_DIR, "reports")

os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,   exist_ok=True)
os.makedirs(REPORTS_DIR,  exist_ok=True)

# ===== OPTIONAL IMPORTS =====
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
    print("✅ PyTorch available — full training enabled")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not found — run: pip install torch torchvision")

try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import pickle
    SKLEARN_AVAILABLE = True
    print("✅ Scikit-learn available — audio training enabled")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  Scikit-learn not found — run: pip install scikit-learn")

try:
    import scipy.io.wavfile as wav
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================
#  1. VIOLENCE DETECTION — RWF-2000 Dataset
# ============================================================
class ViolenceDetector:
    """
    Trains a violence/fight detection model using RWF-2000 dataset.
    Uses optical flow + CNN approach.
    """

    def __init__(self):
        self.dataset_path = os.path.join(DATASETS_DIR, "rwf2000")
        self.model_path   = os.path.join(MODELS_DIR, "violence_model.pkl")
        self.model        = None
        self.labels       = ["NonFight", "Fight"]

    def check_dataset(self):
        """Check if RWF-2000 dataset exists."""
        required = [
            os.path.join(self.dataset_path, "train", "Fight"),
            os.path.join(self.dataset_path, "train", "NonFight"),
        ]
        for path in required:
            if not os.path.exists(path):
                print(f"❌ Missing: {path}")
                print("\n📥 Download RWF-2000 from:")
                print("   https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection")
                print(f"   Extract to: {self.dataset_path}")
                return False
        print(f"✅ RWF-2000 dataset found at {self.dataset_path}")
        return True

    def extract_features(self, video_path, max_frames=30):
        """Extract optical flow features from a video."""
        cap = cv2.VideoCapture(video_path)
        features = []
        prev_gray = None
        frame_count = 0

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (64, 64))
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                features.extend([
                    float(np.mean(mag)),
                    float(np.std(mag)),
                    float(np.max(mag)),
                    float(np.mean(ang)),
                ])

            prev_gray = gray
            frame_count += 1

        cap.release()

        # Pad or truncate to fixed size
        target_size = (max_frames - 1) * 4
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]

        return np.array(features, dtype=np.float32)

    def load_dataset(self):
        """Load all videos and extract features."""
        print("\n📂 Loading RWF-2000 dataset...")
        X, y = [], []
        total = 0

        for split in ["train", "val"]:
            for label_idx, label in enumerate(self.labels):
                folder = os.path.join(self.dataset_path, split, label)
                if not os.path.exists(folder):
                    continue

                videos = [f for f in os.listdir(folder)
                         if f.endswith(('.avi', '.mp4', '.mov'))]
                print(f"   {split}/{label}: {len(videos)} videos")

                for i, video_file in enumerate(videos):
                    video_path = os.path.join(folder, video_file)
                    try:
                        features = self.extract_features(video_path)
                        X.append(features)
                        y.append(label_idx)
                        total += 1
                        if (i+1) % 50 == 0:
                            print(f"   Processed {i+1}/{len(videos)} {label} videos...")
                    except Exception as e:
                        print(f"   ⚠️  Skipped {video_file}: {e}")

        print(f"✅ Loaded {total} videos total")
        return np.array(X), np.array(y)

    def train(self):
        """Train violence detection model."""
        if not self.check_dataset():
            return False

        if not SKLEARN_AVAILABLE:
            print("❌ scikit-learn needed: pip install scikit-learn")
            return False

        print("\n🏋️  Training Violence Detection Model...")
        X, y = self.load_dataset()

        if len(X) == 0:
            print("❌ No videos found in dataset!")
            return False

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
        print("   Training Random Forest classifier...")

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred    = model.predict(X_test)
        accuracy  = accuracy_score(y_test, y_pred)
        print(f"\n✅ Violence Model Accuracy: {accuracy:.1%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.labels))

        # Save model
        import pickle
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"💾 Model saved: {self.model_path}")
        self.model = model
        return True

    def load_model(self):
        """Load pre-trained model."""
        if os.path.exists(self.model_path):
            import pickle
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ Violence model loaded from {self.model_path}")
            return True
        return False

    def predict(self, video_path):
        """Predict if a video contains violence."""
        if self.model is None:
            if not self.load_model():
                return "Unknown", 0.0

        features = self.extract_features(video_path)
        features = features.reshape(1, -1)
        pred     = self.model.predict(features)[0]
        proba    = self.model.predict_proba(features)[0]
        label    = self.labels[pred]
        conf     = float(proba[pred])
        return label, conf

    def predict_frame_sequence(self, frames):
        """
        Predict violence from a sequence of OpenCV frames.
        Use this for real-time detection in ai_engine.py
        """
        if self.model is None:
            if not self.load_model():
                return False, 0.0

        features = []
        prev_gray = None

        for frame in frames:
            frame = cv2.resize(frame, (64, 64))
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                features.extend([
                    float(np.mean(mag)),
                    float(np.std(mag)),
                    float(np.max(mag)),
                    float(np.mean(ang)),
                ])
            prev_gray = gray

        target = (len(frames) - 1) * 4
        if len(features) < target:
            features.extend([0.0] * (target - len(features)))
        else:
            features = features[:target]

        feat_arr = np.array(features).reshape(1, -1)
        pred     = self.model.predict(feat_arr)[0]
        proba    = self.model.predict_proba(feat_arr)[0]
        is_fight = bool(pred == 1)
        conf     = float(proba[pred])
        return is_fight, conf


# ============================================================
#  2. AUDIO CLASSIFICATION — ESC-50 Dataset
# ============================================================
class AudioClassifier:
    """
    Trains audio threat detection using ESC-50 dataset.
    Detects: screams, crying, sirens, gunshots.
    """

    # ESC-50 categories relevant to safety
    THREAT_CATEGORIES = {
        "crying_baby":      "MEDIUM",
        "sneezing":         "LOW",
        "clapping":         "LOW",
        "breathing":        "LOW",
        "coughing":         "LOW",
        "footsteps":        "LOW",
        "laughing":         "LOW",
        "brushing_teeth":   "LOW",
        "snoring":          "LOW",
        "drinking_sipping": "LOW",
        "door_wood_knock":  "LOW",
        "mouse_click":      "LOW",
        "keyboard_typing":  "LOW",
        "door_wood_creaks": "LOW",
        "can_opening":      "LOW",
        "washing_machine":  "LOW",
        "vacuum_cleaner":   "LOW",
        "clock_alarm":      "MEDIUM",
        "clock_tick":       "LOW",
        "glass_breaking":   "HIGH",
        "helicopter":       "LOW",
        "chainsaw":         "HIGH",
        "siren":            "HIGH",
        "car_horn":         "MEDIUM",
        "engine":           "LOW",
        "train":            "LOW",
        "church_bells":     "LOW",
        "airplane":         "LOW",
        "fireworks":        "HIGH",
        "hand_saw":         "MEDIUM",
        "dog":              "LOW",
        "rooster":          "LOW",
        "pig":              "LOW",
        "cow":              "LOW",
        "frog":             "LOW",
        "cat":              "LOW",
        "hen":              "LOW",
        "insects":          "LOW",
        "sheep":            "LOW",
        "crow":             "LOW",
        "rain":             "LOW",
        "sea_waves":        "LOW",
        "crackling_fire":   "MEDIUM",
        "crickets":         "LOW",
        "chirping_birds":   "LOW",
        "water_drops":      "LOW",
        "wind":             "LOW",
        "pouring_water":    "LOW",
        "toilet_flush":     "LOW",
        "thunderstorm":     "LOW",
    }

    def __init__(self):
        self.dataset_path = os.path.join(DATASETS_DIR, "esc50")
        self.model_path   = os.path.join(MODELS_DIR, "audio_model.pkl")
        self.label_path   = os.path.join(MODELS_DIR, "audio_labels.json")
        self.model        = None
        self.label_encoder = None

    def check_dataset(self):
        audio_dir = os.path.join(self.dataset_path, "audio")
        csv_path  = os.path.join(self.dataset_path, "meta", "esc50.csv")

        if not os.path.exists(audio_dir) or not os.path.exists(csv_path):
            print(f"❌ ESC-50 dataset not found!")
            print("\n📥 Download ESC-50 from:")
            print("   https://github.com/karolpiczak/ESC-50")
            print("   Click: Code → Download ZIP")
            print(f"   Extract audio/ and meta/ to: {self.dataset_path}")
            return False

        print(f"✅ ESC-50 dataset found")
        return True

    def extract_features(self, audio_path, sample_rate=22050):
        """Extract audio features from .wav file."""
        try:
            sr, data = wav.read(audio_path)

            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = data.mean(axis=1)

            data = data.astype(np.float32)

            # Normalize
            if np.max(np.abs(data)) > 0:
                data = data / np.max(np.abs(data))

            # Resample to standard rate
            target_length = sample_rate * 5  # 5 seconds
            if len(data) > target_length:
                data = data[:target_length]
            else:
                data = np.pad(data, (0, target_length - len(data)))

            # Feature extraction
            features = []

            # 1. RMS energy
            frame_length = 1024
            hop_length   = 512
            rms_frames   = []
            for i in range(0, len(data) - frame_length, hop_length):
                frame = data[i:i+frame_length]
                rms_frames.append(np.sqrt(np.mean(frame**2)))
            features.extend([
                np.mean(rms_frames), np.std(rms_frames),
                np.max(rms_frames),  np.min(rms_frames)
            ])

            # 2. Zero crossing rate
            zcr = np.sum(np.diff(np.sign(data)) != 0) / len(data)
            features.append(zcr)

            # 3. Spectral features via FFT
            fft  = np.abs(np.fft.rfft(data[:frame_length*4]))
            freqs = np.fft.rfftfreq(frame_length*4, 1.0/sample_rate)

            def band(low, high):
                m = (freqs >= low) & (freqs <= high)
                return float(np.mean(fft[m])) if np.any(m) else 0.0

            features.extend([
                band(0,    300),    # Sub-bass
                band(300,  1000),   # Voice low
                band(1000, 3000),   # Voice high / Scream
                band(3000, 8000),   # High freq
                band(8000, 16000),  # Very high
            ])

            # 4. Spectral centroid
            if np.sum(fft) > 0:
                centroid = np.sum(freqs[:len(fft)] * fft) / np.sum(fft)
            else:
                centroid = 0.0
            features.append(centroid)

            # 5. Volume spikes
            spikes = np.sum(np.array(rms_frames) > np.mean(rms_frames) * 2)
            features.append(spikes)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            print(f"⚠️  Audio feature error {audio_path}: {e}")
            return np.zeros(12, dtype=np.float32)

    def load_dataset(self):
        """Load ESC-50 dataset."""
        import csv
        print("\n📂 Loading ESC-50 dataset...")

        csv_path  = os.path.join(self.dataset_path, "meta", "esc50.csv")
        audio_dir = os.path.join(self.dataset_path, "audio")

        X, y, names = [], [], []
        count = 0

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows   = list(reader)

        print(f"   Total files in CSV: {len(rows)}")

        for row in rows:
            filename = row['filename']
            category = row['category']
            audio_path = os.path.join(audio_dir, filename)

            if not os.path.exists(audio_path):
                continue

            features = self.extract_features(audio_path)
            X.append(features)
            y.append(category)
            names.append(category)
            count += 1

            if count % 100 == 0:
                print(f"   Processed {count}/{len(rows)} files...")

        print(f"✅ Loaded {count} audio files")
        return np.array(X), np.array(y)

    def train(self):
        """Train audio classification model."""
        if not self.check_dataset():
            return False

        if not SKLEARN_AVAILABLE:
            print("❌ scikit-learn needed: pip install scikit-learn")
            return False

        print("\n🏋️  Training Audio Classification Model...")
        X, y = self.load_dataset()

        if len(X) == 0:
            print("❌ No audio files found!")
            return False

        from sklearn.preprocessing import LabelEncoder
        from sklearn.ensemble import RandomForestClassifier

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
        print("   Training Random Forest classifier...")

        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred   = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n✅ Audio Model Accuracy: {accuracy:.1%}")

        # Save model and labels
        import pickle
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)

        labels_data = {
            "classes": list(le.classes_),
            "threat_map": self.THREAT_CATEGORIES
        }
        with open(self.label_path, 'w') as f:
            json.dump(labels_data, f, indent=2)

        print(f"💾 Audio model saved: {self.model_path}")
        self.model        = model
        self.label_encoder = le
        return True

    def load_model(self):
        """Load pre-trained audio model."""
        if os.path.exists(self.model_path) and os.path.exists(self.label_path):
            import pickle
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.label_path, 'r') as f:
                data = json.load(f)
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(data["classes"])
            print(f"✅ Audio model loaded from {self.model_path}")
            return True
        return False

    def predict_threat(self, audio_data, sample_rate=44100):
        """
        Predict threat from raw audio numpy array.
        Use this in ai_engine.py for real-time detection.
        """
        if self.model is None:
            if not self.load_model():
                return "LOW", 0.0, "Unknown"

        # Save temp wav
        temp_path = os.path.join(MODELS_DIR, "_temp_audio.wav")
        try:
            import scipy.io.wavfile as wav
            wav.write(temp_path, sample_rate, (audio_data * 32767).astype(np.int16))
            features = self.extract_features(temp_path, sample_rate)
            features = features.reshape(1, -1)
            pred     = self.model.predict(features)[0]
            proba    = self.model.predict_proba(features)[0]
            label    = self.label_encoder.inverse_transform([pred])[0]
            conf     = float(proba[pred])
            threat   = self.THREAT_CATEGORIES.get(label, "LOW")
            return threat, conf, label
        except Exception as e:
            return "LOW", 0.0, "Unknown"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


# ============================================================
#  3. NIGHT MODE TESTING — ExDark Dataset
# ============================================================
class NightModeTester:
    """Tests and validates night mode enhancement using ExDark dataset."""

    def __init__(self):
        self.dataset_path = os.path.join(DATASETS_DIR, "exdark")

    def check_dataset(self):
        img_dir = os.path.join(self.dataset_path, "images")
        if not os.path.exists(img_dir):
            print(f"❌ ExDark dataset not found!")
            print("\n📥 Download ExDark from:")
            print("   https://github.com/cs-chan/Exclusively-Dark-Image-Dataset")
            print(f"   Extract images/ to: {self.dataset_path}")
            return False
        print(f"✅ ExDark dataset found")
        return True

    def test_enhancement(self, max_images=50):
        """Test night mode enhancement on ExDark images."""
        if not self.check_dataset():
            return

        img_dir = os.path.join(self.dataset_path, "images")
        images  = [f for f in os.listdir(img_dir)
                  if f.endswith(('.jpg','.png','.jpeg'))][:max_images]

        print(f"\n🌙 Testing Night Mode on {len(images)} dark images...")
        scores = []

        for img_file in images:
            img_path = os.path.join(img_dir, img_file)
            img      = cv2.imread(img_path)
            if img is None:
                continue

            # Original brightness
            gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            orig_brightness = float(np.mean(gray_orig))

            # Apply CLAHE enhancement
            lab      = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b  = cv2.split(lab)
            clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = cv2.cvtColor(
                cv2.merge([clahe.apply(l), a, b]),
                cv2.COLOR_LAB2BGR
            )

            # Enhanced brightness
            gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            enh_brightness = float(np.mean(gray_enh))

            improvement = enh_brightness - orig_brightness
            scores.append(improvement)

        if scores:
            avg_improvement = np.mean(scores)
            print(f"✅ Night Mode Test Complete")
            print(f"   Average brightness improvement: +{avg_improvement:.1f}")
            print(f"   Images tested: {len(scores)}")
            print(f"   Night mode works correctly!" if avg_improvement > 5 else
                  f"   Night mode needs tuning")

        return scores


# ============================================================
#  4. LOITERING/TRACKING — MOT17 Dataset
# ============================================================
class LoiteringTester:
    """Tests loitering detection using MOT17 tracking dataset."""

    def __init__(self):
        self.dataset_path = os.path.join(DATASETS_DIR, "mot17")

    def check_dataset(self):
        train_dir = os.path.join(self.dataset_path, "train")
        if not os.path.exists(train_dir):
            print(f"❌ MOT17 dataset not found!")
            print("\n📥 Download MOT17 from:")
            print("   https://motchallenge.net/data/MOT17/")
            print(f"   Extract to: {self.dataset_path}")
            return False
        print(f"✅ MOT17 dataset found")
        return True

    def test_tracking(self):
        """Test person tracking and loitering detection."""
        if not self.check_dataset():
            return

        train_dir  = os.path.join(self.dataset_path, "train")
        sequences  = [d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d))]

        if not sequences:
            print("❌ No sequences found in MOT17")
            return

        seq_path   = os.path.join(train_dir, sequences[0], "img1")
        if not os.path.exists(seq_path):
            print(f"❌ Image folder not found: {seq_path}")
            return

        images     = sorted([f for f in os.listdir(seq_path)
                            if f.endswith('.jpg')])[:100]

        print(f"\n🚶 Testing Loitering Detection on MOT17/{sequences[0]}")
        print(f"   Processing {len(images)} frames...")

        # Simple centroid tracker
        person_tracker = {}
        frame_count    = 0

        for img_file in images:
            img_path = os.path.join(seq_path, img_file)
            frame    = cv2.imread(img_path)
            if frame is None:
                continue

            frame_count += 1

            # HOG person detection
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects, _ = hog.detectMultiScale(gray, winStride=(8,8), padding=(4,4))

            for (x,y,w,h) in rects:
                cx, cy = x + w//2, y + h//2
                pid    = f"{cx//50}_{cy//50}"  # Simple grid ID

                if pid not in person_tracker:
                    person_tracker[pid] = {"first_frame": frame_count, "count": 0}
                person_tracker[pid]["count"] += 1

        # Check loitering (person in many frames)
        loiterers = {
            pid: data for pid, data in person_tracker.items()
            if data["count"] > 20
        }

        print(f"✅ Tracking Test Complete")
        print(f"   Frames processed : {frame_count}")
        print(f"   Persons tracked  : {len(person_tracker)}")
        print(f"   Loiterers found  : {len(loiterers)}")


# ============================================================
#  5. MODEL INTEGRATION — Add to ai_engine.py
# ============================================================
class ModelIntegration:
    """
    Shows how to integrate trained models into ai_engine.py.
    Prints the exact code to add.
    """

    def show_integration_code(self):
        print("\n" + "="*60)
        print("HOW TO ADD TRAINED MODELS TO ai_engine.py")
        print("="*60)
        print("""
1. VIOLENCE DETECTION - Add to SafeWatchDetector.__init__():

    from dataset_trainer import ViolenceDetector
    self.violence_detector = ViolenceDetector()
    self.violence_detector.load_model()
    self.frame_buffer = []  # Buffer for violence detection

2. VIOLENCE DETECTION - Add to process_video_stream() loop:

    # Add frame to buffer
    detector.frame_buffer.append(frame)
    if len(detector.frame_buffer) > 30:
        detector.frame_buffer.pop(0)

    # Check violence every 15 frames
    if frame_num % 15 == 0 and len(detector.frame_buffer) >= 10:
        is_fight, conf = detector.violence_detector.predict_frame_sequence(
            detector.frame_buffer[-10:]
        )
        if is_fight and conf > 0.7:
            print(f"FIGHT DETECTED! Confidence: {conf:.0%}")

3. AUDIO CLASSIFICATION - Add to AudioAnalyzer._analyze():

    from dataset_trainer import AudioClassifier
    self.audio_classifier = AudioClassifier()
    self.audio_classifier.load_model()

    # In _analyze method, add:
    if hasattr(self, 'audio_classifier') and self.audio_classifier.model:
        threat, conf, label = self.audio_classifier.predict_threat(audio_data)
        # Use this instead of the manual classification
""")


# ============================================================
#  MAIN — Run all training tasks
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SafeWatch Dataset Trainer")
    parser.add_argument("--task", default="check",
        choices=["check", "violence", "audio", "night", "tracking", "all", "integrate"],
        help="Task to run")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  SafeWatch Dataset Trainer")
    print("="*60 + "\n")

    if args.task == "check" or args.task == "all":
        print("📋 Checking datasets...\n")
        ViolenceDetector().check_dataset()
        AudioClassifier().check_dataset()
        NightModeTester().check_dataset()
        LoiteringTester().check_dataset()

    if args.task == "violence" or args.task == "all":
        vd = ViolenceDetector()
        vd.train()

    if args.task == "audio" or args.task == "all":
        ac = AudioClassifier()
        ac.train()

    if args.task == "night" or args.task == "all":
        nt = NightModeTester()
        nt.test_enhancement()

    if args.task == "tracking" or args.task == "all":
        lt = LoiteringTester()
        lt.test_tracking()

    if args.task == "integrate":
        ModelIntegration().show_integration_code()

    print("\n✅ Done!")
    print(f"📁 Trained models saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()