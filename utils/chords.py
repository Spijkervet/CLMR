chromatic_scale = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}

fifths = {
    "C": "G",
    "C#": "G#",
    "D": "A",
    "D#": "A#",
    "E": "B",
    "F": "C",
    "F#": "C#",
    "G": "D",
    "G#": "D#",
    "A": "E",
    "A#": "F",
    "B": "F#",
}

chords = {
    "C:maj": 0,
    "C#:maj": 1,
    "D:maj": 2,
    "D#:maj": 3,
    "E:maj": 4,
    "F:maj": 5,
    "F#:maj": 6,
    "G:maj": 7,
    "G#:maj": 8,
    "A:maj": 9,
    "A#:maj": 10,
    "B:maj": 11,
    "C:min": 12,
    "C#:min": 13,
    "D:min": 14,
    "D#:min": 15,
    "E:min": 16,
    "F:min": 17,
    "F#:min": 18,
    "G:min": 19,
    "G#:min": 20,
    "A:min": 21,
    "A#:min": 22,
    "B:min": 23,
    "N": 24,
    "X": 25,
}

keys = {
    "C:maj": 0,
    "C#:maj": 1,
    "D:maj": 2,
    "D#:maj": 3,
    "E:maj": 4,
    "F:maj": 5,
    "F#:maj": 6,
    "G:maj": 7,
    "G#:maj": 8,
    "A:maj": 9,
    "A#:maj": 10,
    "B:maj": 11,
    "C:min": 12,
    "C#:min": 13,
    "D:min": 14,
    "D#:min": 15,
    "E:min": 16,
    "F:min": 17,
    "F#:min": 18,
    "G:min": 19,
    "G#:min": 20,
    "A:min": 21,
    "A#:min": 22,
    "B:min": 23,
}


def enharmonic(chord):
    new_chord = chord
    enharmonic_table = {
        "Cb": "B",
        "Db": "C#",
        "Eb": "D#",
        "Fb": "E",
        "Gb": "F#",
        "Ab": "G#",
        "Bb": "A#",
    }

    try:
        new_chord = enharmonic_table[chord]
    except:
        print("Enharmonic root not found: {}".format(chord))

    return new_chord


def get_chord_at_interval(chord_data, start_idx, end_idx, samplerate):
    # convert samples to seconds
    start_sec = start_idx / samplerate
    end_sec = end_idx / samplerate
    chords = []
    for idx, intervals in enumerate(chord_data.intervals):
        if start_sec >= intervals[0] and start_sec <= intervals[1]:
            chords.append([chord_data.labels[idx], intervals[1]-intervals[0]])
            for idx, intervals in enumerate(chord_data.intervals):
                if end_sec >= intervals[0] and end_sec <= intervals[1]:
                    chords.append([chord_data.labels[idx], intervals[1]-intervals[0]])
                    break
    
    chords_sorted = sorted(chords, key=lambda x: x[1], reverse=True)
    return [chords_sorted[0][0]] # get most frequent chord, "majority vote"
    # return list(set(chords))


def key_to_label(key):
    tonic, mode = key.split(":")
    if "b" in tonic:
        tonic = enharmonic(tonic)
        key = tonic + ":" + mode
    try:
        return keys[key]
    except:
        print("Key missing: {}".format(key))

def label_to_key(label):
    for k, v in keys.items():
        if label == v:
            return k
    return None

def mode_to_label(mode):
    if mode == "min":
        return 0
    if mode == "maj":
        return 1
    return None


def chord_to_label(chord):
    root = chord.split(":")[0]
    if "b" in root:
        quality = chord.split(":")[1]
        chord = enharmonic(root) + ":" + quality
    try:
        return chords[chord]
    except:
        print("Chord missing: {}".format(chord))

def to_one_hot(labels, num_classes):
    vec = np.zeros(num_classes)


def estimate_mode(tonic, chords):
    labels = chords.labels
    labels = [m for m in labels if len(m) > 1 and (m != "N" and m != "X")]  # discard N

    num_tonic = 0
    num_major = 0
    num_minor = 0
    total_chords = len(labels)
    for l in labels:
        if ":" not in l:
            root = l
            mode = "maj"
        else:
            root, mode = l.split(":")

        root = ''.join([i for i in root if not i.isdigit()])
        root = root.replace('/', '')

        # force to major
        if "maj" in mode:
            mode = "maj"
        elif "min" in mode:
            mode = "min"

        if mode in ["aug", "hdim7", "7", "9", "sus4", "(1)"]:
            mode = "maj"
        elif mode in ["min/b3", "dim"]:
            mode = "min"

        if root == tonic:
            num_tonic += 1

            if mode == "maj":
                num_major += 1
            elif mode == "min":
                num_minor += 1

    # discard song if no tonic is found
    if num_tonic == 0:
        return None

    # 1. select all chords whose root is the tonic
    # 2. if more than 90% of these chords are major, the key mode is assumed to be major, and vice-versa for minor;
    # 3. else, discard the song, because we cannot confidently estimate the mode.
    perc_major = num_major / num_tonic
    perc_minor = num_minor / num_tonic

    if perc_minor >= 0.9:
        return "min"  # minor
    if perc_major >= 0.9:
        return "maj"  # major

    return None
