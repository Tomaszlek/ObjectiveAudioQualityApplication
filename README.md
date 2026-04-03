# Objective Audio Quality Assessment Application

Desktop application for audio quality evaluation using MATLAB algorithms (PEAQ, VisqolA) and PyTorch deep learning models.

## Overview

The application provides a three-stage workflow for assessing audio quality:
1. **Preprocessing** - Fragment selection and audio loading
2. **Subjective Assessment** - Manual audio playback and rating
3. **Objective Analysis** - Automated analysis using MATLAB and ML models

## Technology Stack

- **GUI Framework**: PyQt6 with dark_teal theme
- **Audio Processing**: librosa, scipy, soundfile, sounddevice
- **Deep Learning**: PyTorch 2.7, TorchVision 0.22
- **MATLAB Integration**: MATLAB Engine for Python 24.2
- **Database**: SQLite
- **Visualization**: pyqtgraph

## Installation

### Prerequisites
- Python 3.9+
- MATLAB R2024a or later with Engine for Python
- FFmpeg

### Steps

1. Clone the repository
```bash
git clone https://github.com/Tomaszlek/ObjectiveAudioQualityApplication.git
cd ObjectiveAudioQualityApplication
```

2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/macOS
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
python app/main.py
```

## Application Features

### Stage 1: Preprocessing (PreprocessingView)
- Load audio files with pyqtgraph waveform visualization
- Select audio fragments (7-second fixed duration)
- Linear region selection on waveform
- Fragment search via worker thread
- Audio playback with sounddevice

### Stage 2: Subjective Assessment (SubjectiveView)
- Display project data in table format
- Side-by-side audio playback (reference vs degraded)
- pyqtgraph waveform visualization for both channels
- Audio caching for performance
- Manual rating and notes

### Stage 3: Objective Analysis (ObjectiveView)
- Single file processing via SingleFileProcessorWorker
- Batch processing via AnalysisWorker
- MATLAB-based analysis (PEAQ, VisqolA)
- PyTorch model inference
- Results table display
- Context menu for file operations

## Audio Processing Pipeline

### AudioProcessor
- Loads audio using librosa (48 kHz mono)
- Signal alignment using cross-correlation
- Calls MATLAB functions:
  - `runVisqolForPair()` - VisqolA analysis
  - `PEAQTest()` - PEAQ analysis
- Temporary file cleanup

### PyTorchProcessor
Loads and runs 4 pre-trained CNN models:

| Model | File | Normalization | Input Size |
|-------|------|----------------|-----------|
| CNN 1D | cnn_1d_unipolared.pth | unipolar | 256×657 |
| InceptionV3 | inception_v3_unipolared.pth | unipolar | 299×299 |
| VGG19 | vgg19_bipolared.pth | bipolar | 224×224 |
| EfficientNet V2-S | efficientnet_v2_s_bipolared.pth | bipolar | 384×384 |

Features:
- Mel-spectrogram generation (2048 FFT, 256 hop length, 256 mel bins)
- Automatic GPU/CPU detection
- Model weight caching
- Unipolar [0, 1] and bipolar [-1, 1] normalization
- Batch inference support

## MATLAB Integration

### Configuration
Edit `app/config.ini`:
```ini
[MATLAB_PATHS]
peaq_root = matlab_scripts/PEAQ
visqol_root = matlab_scripts/VisqolA
pqeval_root = matlab_scripts/PEAQ/PQevalAudio
pqeval_cb = matlab_scripts/PEAQ/PQevalAudio/CB
pqeval_misc = matlab_scripts/PEAQ/PQevalAudio/Misc
pqeval_mov = matlab_scripts/PEAQ/PQevalAudio/MOV
pqeval_patt = matlab_scripts/PEAQ/PQevalAudio/Patt
```

### MATLAB Scripts
- `matlab_scripts/PEAQ/` - Perceptual Evaluation of Audio Quality algorithm
- `matlab_scripts/VisqolA/` - Google's Audio Quality metrics

## Database

SQLite database at `output/projekt.db` stores:
- `audio_pairs` table with fields:
  - `id`, `ref_path`, `deg_path`, `bitrate`, `noise_level`, `filter_cutoff`
  - `mos_lqo` (VisqolA score), `odg` (PEAQ ODG)
  - `user_rating`, `timestamp`

Operations via `DatabaseManager`:
- Load/save audio pairs
- Query results as pandas DataFrame
- Insert/update analysis results

## Project Structure

```
ObjectiveAudioQualityApplication/
├── app/
│   ├── main.py                    # Entry point, applies dark_teal theme
│   ├── config.ini                 # MATLAB paths configuration
│   ├── Controllers/
│   │   └── app_controller.py      # Orchestrates MATLAB engine, views, database
│   ├── Models/
│   │   ├── audio_processor.py     # MATLAB audio analysis interface
│   │   ├── pytorch_processor.py   # PyTorch model inference
│   │   ├── model_architecture.py  # 4 CNN model definitions
│   │   └── database_manager.py    # SQLite database operations
│   ├── Views/
│   │   ├── main_window.py         # Main UI (loads main_window.ui)
│   │   ├── preprocessing/         # Stage 1 preprocessing interface
│   │   ├── playback/              # Stage 2 subjective assessment interface
│   │   └── objective/             # Stage 3 objective analysis interface
│   ├── Services/
│   │   └── file_generation_service.py  # File export operations
│   ├── Workers/
│   │   ├── worker.py              # Base AnalysisWorker thread
│   │   ├── single_file_worker.py  # Single file analysis thread
│   │   └── find_fragment_worker.py # Fragment search thread
│   └── matlab_scripts/
│       ├── PEAQ/                  # PEAQ algorithm with subfolder structure
│       └── VisqolA/               # VisqolA algorithm
├── models/                        # Pre-trained PyTorch weights
│   ├── cnn_1d_unipolared.pth
│   ├── inception_v3_unipolared.pth
│   ├── vgg19_bipolared.pth
│   └── efficientnet_v2_s_bipolared.pth
├── training_notebook/             # Jupyter notebook for model training
│   └── w-asna-metoda-obiektywnej-oceny-jako-ci-d-wi-ku-u.ipynb
├── results/                       # Training analysis outputs
│   ├── 1_korelacje_podstawowe.png
│   ├── 2_korelacje_z_przedzialami.png
│   ├── 3_boxplot_bledow.png
│   ├── KOMPLET_Histogram_FINAL.png
│   ├── KOMPLET_Reszty_FINAL.png
│   └── tabela_wynikow.xlsx
├── output/                        # Application output directory (created at runtime)
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Model Architectures

### CNN 1D
- Input: 256×657 unipolar spectrogram
- 3 conv layers (kernel sizes: 11, 7, 3) with ReLU and MaxPool
- 3 fully connected layers (256→128→1)
- Output: Single quality score

### InceptionV3
- Input: 1-channel image → resized to 299×299
- Backbone: ImageNet pre-trained InceptionV3 (frozen)
- Custom head: 3 fully connected layers
- Unipolar normalization

### VGG19
- Input: 1-channel image → resized to 224×224
- Backbone: ImageNet pre-trained VGG19 features (frozen)
- Custom head: Flatten + 3 fully connected layers
- Bipolar normalization

### EfficientNet V2-S
- Input: 1-channel image → resized to 384×384
- Backbone: ImageNet pre-trained EfficientNet V2-S (frozen)
- Custom classifier: Flatten + 3 fully connected layers
- Bipolar normalization

## Jupyter Notebook

**File**: `training_notebook/w-asna-metoda-obiektywnej-oceny-jako-ci-d-wi-ku-u.ipynb`

Contains model training pipeline with:
- Dataset loading and preprocessing
- 4 CNN model architectures
- Training loop with early stopping
- Evaluation metrics (Pearson, Spearman, RMSE, MAE, R²)
- Visualization outputs (correlations, boxplots, histograms)
- Per-model results in Excel format

To run:
```bash
pip install jupyter
jupyter notebook training_notebook/
```

## Usage Workflow

1. **Launch Application**
   ```bash
   python app/main.py
   ```
   MATLAB engine starts automatically (may take several seconds)

2. **Preprocessing Stage**
   - Load audio files
   - Select 7-second fragments
   - Preview waveforms

3. **Subjective Stage**
   - Play reference and degraded audio
   - Provide manual quality ratings

4. **Objective Stage**
   - Run MATLAB analysis (PEAQ + VisqolA)
   - Run selected PyTorch models
   - View results in table
   - Export to database

5. **Review Results**
   - Access `output/projekt.db`
   - Export to CSV via DatabaseManager

## Performance Notes

- Audio normalization: subsecond for 10s audio
- Signal alignment (cross-correlation): ~1-2s
- MATLAB PEAQ analysis: ~5-8s per pair
- PyTorch inference (all 4 models): GPU accelerated
- UI updates via worker threads to prevent freezing

## Dependencies

Key packages from `requirements.txt`:
- PyQt6 (GUI)
- PyTorch + TorchVision (ML models)
- librosa (audio processing)
- scipy (signal processing)
- soundfile + sounddevice (audio I/O)
- matlabengine (MATLAB integration)
- pandas (data management)
- pyqtgraph (visualization)
