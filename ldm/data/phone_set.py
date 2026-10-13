"""Phoneme / note vocabulary shared by the datasets and the models.

The phoneme list is read from phone_set.json (written by preprocess/gtsinger_build_metadata.py);
set TCSINGER2_PHONE_SET to use another file. A checkpoint is tied to the phone set it was trained with.
"""
import json
import os

_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHONE_SET_PATH = os.environ.get('TCSINGER2_PHONE_SET', os.path.join(_REPO_DIR, 'data', 'meta', 'phone_set.json'))

if not os.path.exists(PHONE_SET_PATH):
    raise FileNotFoundError(f'phone set not found: {PHONE_SET_PATH} (build it with '
                            'preprocess/gtsinger_build_metadata.py or set TCSINGER2_PHONE_SET)')
with open(PHONE_SET_PATH, encoding='utf-8') as f:
    ph_set = json.load(f)

ph2id = {ph: idx for idx, ph in enumerate(ph_set)}
PH_PAD_ID = len(ph_set)       # padding / masked-boundary token
NUM_PH = len(ph_set) + 1      # embedding size

NUM_MIDI = 100                # note pitches 0 (rest) .. 98
MIDI_PAD_ID = 99
NOTE_TYPE_PAD = 4             # note types: 1 rest, 2 single-note word, 3 multi-note word

language_set = ['Chinese', 'English', 'French', 'German', 'Italian', 'Japanese', 'Korean', 'Russian', 'Spanish']
NUM_LANG = len(language_set)
