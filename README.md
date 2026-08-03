# BOSSA: Learning Music Style For Piano Arrangement Through Cross-Modal Bootstrapping

 Repository for *J. Zhao, G. Xia, Z. Wang, and Y. Wang. Learning Music Style For Piano Arrangement Through Cross-Modal Bootstrapping, in Proceedings of ISMIR 2026*.

- 📄 **Paper**: tbd
- 🎧 **Demo page**: https://zhaojw1998.github.io/bossa/
- 💾 **Model checkpoints**: [Google Drive (`checkpoints.tar.gz`)](https://drive.google.com/file/d/1DUzr670TXi1WK1xrIJadigZqY6KRlaro/view?usp=sharing)

## Overview

Music style is usually described with text labels ("swing", "classical", "emotional"), but the real style is
implicit and only exists in concrete music examples. BOSSA learns style directly from **raw audio** and applies
it to **symbolic** piano arrangement: given a *lead sheet* (melody + chords as content)
and a *reference audio* (which provides the style), the model generates an expressive piano cover in MIDI.

## File structure

```
.
├── model_blip2.py            # Blip2Qformer (stage I) and Blip2Musecoco (stage II); audio encoder init
├── model_Qformer.py          # BERT/Q-Former backbone with cross-attention (adapted from LAVIS)
├── dataset.py                # Ensemble (PIAST + POP909) dataset, collate fns for training & inference
├── train_stage1.py           # Stage-I contrastive/generative pre-training (DDP)
├── train_stage2.py           # Stage-II arrangement training with LoRA (DDP)
├── test_stage1_contrastive.py  # ⭐ Task 1: audio-to-MIDI retrieval with the stage-I model
├── test_stage2_generate.ipynb  # ⭐ Task 2: piano cover generation with the stage-II model
├── sheetsage_inference.py    # Batch version of task 2 over folders of audio × lead sheets
├── requirements.txt
│
├── musecoco/                 # Symbolic music LM (MuseCoco), HuggingFace port
│
├── data_processing/          # Offline corpus preparation (only needed for training)
│
└── utils/                    # Other utility functions
```

## Environment

Tested with Python 3.10 and CUDA 11.8+. A GPU with ≥24 GB memory is recommended for inference
(MusicGen-large and MuseCoco-1B are both resident).

```bash
conda create -n bossa python=3.10
conda activate bossa

# install torch/torchaudio， tested for version 2.6.0
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

## Checkpoints

Download [`checkpoints.tar.gz`](https://drive.google.com/file/d/1DUzr670TXi1WK1xrIJadigZqY6KRlaro/view?usp=sharing)
and decompress it in the repository root. The weights are then expected at the following locations:

```
.
├── musecoco/
│   └── 1b/
│       └── model/
│           ├── config.json                       # already in the repo
│           └── model.safetensors                 # ← MuseCoco 1B backbone (required, fixed location)
└── checkpoints/
    ├── bossa_stage1.pt                           # ← stage-I Q-Former (retrieval; stage-II init)
    ├── bossa_stage2.pt                           # ← stage-II BOSSA model (piano cover generation)
    └── checkpoint_last_musicbert_base.pt         # ← MusicBERT base (only for stage-I training)
```

## Inference

BOSSA supports two inference tasks, one per stage.

### Task 1 — Audio-to-MIDI retrieval (stage-I model)

The stage-I Q-Former aligns auditory and symbolic style, so its embeddings can be used to retrieve the piano
MIDI that stylistically matches a given audio clip (and vice versa). Run:

```bash
python test_stage1_contrastive.py
```

In [`test_stage1_contrastive.py`](test_stage1_contrastive.py) set:

- the test set — `PIAST_Dataset` / `Ensemble_Dataset` with your preprocessed `.h5` codec file and OctupleMIDI folder (see [Training](#training) for how those are produced);
- `CUDA_VISIBLE_DEVICES` at the bottom of the file.

The script calls [`Blip2Qformer.test_contrastive`](model_blip2.py#L120) on each batch, which ranks every MIDI
candidate in the batch against each audio query, and reports the **mean rank** of the ground-truth pair

### Task 2 — Piano cover generation (stage-II model)

Generate a piano cover whose **content** comes from a lead sheet and whose **style** comes from a reference
audio clip. The main entry point is [`test_stage2_generate.ipynb`](test_stage2_generate.ipynb).

Inputs:
1. **Reference audio** (`.mp3`/`.wav`) — the style source
2. **Lead sheet MIDI** — the content source. It must contain exactly two tracks, in order:
   **(1) melody** and **(2) chords**, with the chord track voiced around C3–C4. See
   [`read_leadsheet`](utils/inference.py#L44) for the exact expectations.

Steps in the notebook:

1. Set `os.environ['CUDA_VISIBLE_DEVICES']` and `DEVICE`.
2. Load the EnCodec audio tokenizer from `facebook/musicgen-small`.
3. Instantiate `Blip2Musecoco()` and load the stage-II checkpoint downloaded above.
4. Fill in the "set audio and leadsheet path" cell:
   - `audio_path`, `midi_path`, `audio_start_time` (seconds into the reference audio)
   - `STAR_BAR` / `DURATION` — the bar range of the lead sheet to arrange
   - `HOP_LEN = 2`, `BAR_LEN = 4` — the sliding window (leave as-is unless you retrain)
   - `save_path` — output directory
5. Run the generation cell.

Generation rolls over the piece in **4-bar windows with a 2-bar hop**, feeding the previous window's last two
bars back in as a prefix so consecutive windows stay coherent.

For batch generation over a folder of reference audio × a folder of lead sheets, use
[`sheetsage_inference.py`](sheetsage_inference.py) — same pipeline in a loop, plus
`read_midi_by_sheetsage()` for lead sheets transcribed by
[SheetSage](https://github.com/chrisdonahue/sheetsage) (beat track first, then melody and chords).

## Training

Training data is PIAST (piano covers) and POP909, both preprocessed offline into
(EnCodec tokens, OctupleMIDI note events) pairs by the scripts in [`data_processing/`](data_processing/). Once the paths in
[`dataset.py`](dataset.py) point at the processed `.h5` / MIDI folders:

```bash
python train_stage1.py              # contrastive + matching + LM pre-training of the Q-Former
python train_stage2.py              # arrangement training; set stage_1_checkpoint to the stage-I .pt
```


## Acknowledgements

This work is builds upon [Musecoco's huggingface implementation]( https://github.com/Sonata165/UnofficialMuseCoco) generously shared by Dr. Longshen Ou.

We also build on [BLIP-2](https://github.com/salesforce/LAVIS) for the Q-Former architecture and
two-stage bootstrapping recipe, [MusicGen](https://github.com/facebookresearch/audiocraft) for the
audio language model, and
[MusicBERT](https://github.com/microsoft/muzic/tree/main/musicbert) for the Q-Former initialization. 

Our models
are trained on the [PIAST](https://hayeonbang.github.io/PIAST_dataset/) and
[POP909](https://github.com/music-x-lab/POP909-Dataset) datasets.

