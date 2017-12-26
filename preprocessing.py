import sys
from functools import lru_cache

import numpy as np
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tqdm import tqdm


IDX_SLUR = 0
IDX_BEAT = 2


@lru_cache(maxsize=1)
def get_score_names():
    return np.load('./data/score_names.npy')


@lru_cache(maxsize=1)
def get_score_tensors():
    print('Loading score tensors from disk')
    sys.stdout.flush()
    score_names = get_score_names()
    return [
        (np.load('./data/{}.npy'.format(i)))
        for i in tqdm(range(len(score_names)))
    ]


@lru_cache(maxsize=1)
def get_metadata_tensors():
    print('Loading metadata tensors from disk')
    sys.stdout.flush()
    score_names = get_score_names()
    return [
        (np.load('./data/{}_meta.npy'.format(i)))
        for i in tqdm(range(len(score_names)))
    ]


def get_train_test(test_size=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    scores = get_score_tensors()
    meta = get_metadata_tensors()
    return train_test_split(scores, meta, test_size=test_size)


def make_targets(score, voice, n_notes, min_pitch):
    """
    Make target variable for notes and rests
    """
    n_output_features = n_notes + 1
    y = np.zeros((score.shape[1], n_output_features))  # shape: n timesteps X m features
    for i, note in enumerate(score[voice]):
        if note > 0:
            note_idx = int(note - min_pitch)
            y[i, note_idx + 1] = 1
        else:
            y[i, 0] = 1  # it's a rest
    return y


def make_targets_slur(meta, voice):
    """
    Make target variable for the slur metadata.
    """
    return meta[voice, :, IDX_SLUR]


def make_padded(score, window_size, max_voices=None):
    # pad the beginning of the sequence so that our first window ends on the first time step
    padding_size = window_size - 1

    score_padding = np.zeros((score.shape[0], padding_size))
    return np.hstack((score_padding, score))


def make_input_beat(meta, voice):
    return meta[voice, :, IDX_BEAT:]


def make_input_sequence(score, meta, voice, max_pitch, sequence_steps=4, conv_window_size=32):
    """
    Make an input sequence for a particular voice
    """
    window_size = sequence_steps * conv_window_size
    # First, do the notes channel
    padded_score = make_padded(score, window_size) / max_pitch
    padding_size = window_size - 1

    # Now, the slurs channel
    padded_meta = make_padded(meta[:, :, 0], window_size)

    # A mask showing which voice to predict
    voice_mask = np.zeros(padded_meta.shape)

    # Stack them together
    indexer = np.arange(window_size)[None, :] + np.arange(
        padded_score.shape[1] - padding_size)[:, None]
    stacked = np.stack((padded_score, padded_meta, voice_mask), axis=-1)

    # Make the sliding windows
    sequence = stacked.swapaxes(0, 1)[indexer, :, :]

    # Now, mask out the target values
    sequence[:, -1, voice, :2] = 0

    # Set a flag in the voice mask to indicate which voice is to be predicted
    sequence[:, -1, voice, 2] = 1

    return sequence.reshape((score.shape[1], -1, conv_window_size, padded_score.shape[0], 3))


class Batch(Sequence):
    def __init__(self, scores, meta, sequence_steps=4, window_size=32, subsample_voices=False):
        self.scores = scores
        self.meta = meta

        self.max_pitch = np.max([np.max(t) for t in scores])
        self.min_pitch = np.min([np.min(t[t > 0]) for t in scores])

        self.n_notes = int(self.max_pitch - self.min_pitch) + 1
        self.sequence_steps = sequence_steps
        self.window_size = window_size

        if subsample_voices:
            # Take one randomly sampled voice for each score
            voice_sample = [
                np.random.randint(score.shape[0])
                for score in scores
            ]
            self.indices = [
                (score_idx, voice_sample[score_idx])
                for score_idx, score in enumerate(scores)
            ]
        else:
            self.indices = [
                (score_idx, voice_idx)
                for score_idx, score in enumerate(scores)
                for voice_idx in range(score.shape[0])
            ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        score_idx, voice = self.indices[idx]
        score = self.scores[score_idx]
        meta = self.meta[score_idx]
        return (
            [
                make_input_sequence(score, meta, voice, self.max_pitch,
                                    sequence_steps=self.sequence_steps,
                                    window_size=self.window_size),
                make_input_beat(meta, voice)
            ],
            [
                make_targets(score, voice, self.n_notes, self.min_pitch),
                make_targets_slur(meta, voice)
            ]
        )

    def on_epoch_end(self):
        pass
