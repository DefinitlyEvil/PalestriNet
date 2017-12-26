from typing import Iterator, Generator

import music21
import numpy as np


# These are by far the most prevalent keys in the corpus, so we're going to restrict ourselves to
# pieces in these keys
KEYS = ['A', 'C', 'D', 'F', 'G']


# Likewise, most pieces have 4, 5, or 6 voices
VOICES = [4, 5, 6]


PIECES = music21.corpus.getComposer('palestrina')


idx_beat = 2
idx_slur = 0
idx_rest = 1
max_beats_per_measure = 16
n_meta_features = 18


def load_pieces_generator(pieces: [str]=PIECES) -> Iterator[music21.stream.Score]:
    return (music21.corpus.parse(piece) for piece in pieces)


def n_voices(score: music21.stream.Score) -> int:
    return len(score.getElementsByClass(music21.stream.Part))


def get_key(score: music21.stream.Score) -> str:
    return score.analyze('key').tonic.fullName


def should_include(score: music21.stream.Score, keys: [str]=KEYS, voices: [int]=VOICES) -> bool:
    return get_key(score) in keys and n_voices(score) in voices


def transpose_to_all_keys_gen(score: music21.stream.Score, keys: [str]=KEYS) \
        -> Generator[music21.stream.Score, None, None]:
    for key in keys:
        score_key = score.analyze('key')
        if score_key.tonic.fullName == key:
            yield score
        else:
            interval = music21.interval.Interval(score_key.tonic, music21.pitch.Pitch(key))
            yield score.transpose(interval)


def get_score_shape(score: music21.stream.Score) -> (int, int):
    n_voices: int = len(score.getElementsByClass(music21.stream.Part))
    n_eighth_notes: int = int(score.duration.quarterLength * 2)
    return n_voices, n_eighth_notes


def score_to_tensor(score: music21.stream.Score) -> (np.ndarray, np.ndarray):
    n_voices, n_eighths = get_score_shape(score)
    score_tensor = np.zeros((n_voices, n_eighths))
    meta_tensor = np.zeros((n_voices, n_eighths, n_meta_features))
    max_beats_per_measure
    try:
        for i, part in enumerate(score.getElementsByClass(music21.stream.Part)):
            for measure in part.getElementsByClass(music21.stream.Measure):
                # we're going to multiply all durations by two,
                # because eighth note is the shortest in the corpus.
                beats_in_measure = measure.duration.quarterLength * 2
                # Get the offset of the beginning of the measure (from the beginning of the piece)
                measure_offset = int(measure.offset)
                for b in range(int(beats_in_measure)):
                    # Annotate each eighth-note pulse in the metadata track
                    meta_tensor[i][measure_offset * 2 + b][idx_beat + b] = 1
                for note in measure.getElementsByClass(music21.note.Note):
                    offset = int(note.offset + measure_offset) * 2
                    for j in range(int(offset), int(offset + note.duration.quarterLength * 2)):
                        # mark the note with its midi pitch throughout its duration
                        score_tensor[i, j] = float(note.midi)
                        if j > offset:
                            # Add a 'slur' annotation for any held note
                            meta_tensor[i, j, idx_slur] = 1
                for rest in measure.getElementsByClass(music21.note.Rest):
                    # Mark all rests in the metadata track
                    offset = int(rest.offset + measure_offset) * 2
                    for j in range(int(offset), int(offset + rest.duration.quarterLength * 2)):
                        meta_tensor[i, j, idx_rest] = 1
        return score_tensor, meta_tensor
    except:
        return None


def tensor_to_score(score_tensor, meta_tensor):
    # TODO: this seems to create accurate output when played as a midi,
    # but doesn't render the rhythms correctly in musescore.
    beats = np.argmax(meta_tensor[:, :, idx_beat:], axis=2)/2
    print(beats)
    measure_length = np.max(beats) + 0.5
    score = music21.stream.Score()
    n_parts, n_beats = score_tensor.shape
    measure = None
    for i in range(n_parts):
        part = music21.stream.Part()
        score.insert(0, part)
        for j in range(n_beats):
            if beats[i, j] == 0:
                measure = music21.stream.Measure()
                part.insert(j / 2, measure)
            if score_tensor[i, j] != 0 and meta_tensor[i, j, idx_slur] == 0:
                duration = 0.5
                k = 1
                while j + k < n_beats and meta_tensor[i, j + k, idx_slur] == 1:
                    duration += 0.5
                    k += 1
                note = music21.note.Note(score_tensor[i, j])
                note.duration.quarterLength = duration
                measure.insert(beats[i, j], note)
            elif score_tensor[i, j] == 0 and (j == 0 or score_tensor[i, j - 1] != 0):
                # insert a rest
                duration = 0.5
                k = 1
                while j + k < n_beats and score_tensor[i, j + k] == 0:
                    duration += 0.5
                    k += 1
                rest = music21.note.Rest()
                rest.duration.quarterLength = duration
                measure.insert(beats[i, j], rest)
    return score
