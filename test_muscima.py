import os
import numpy as np
import itertools
from muscima.io import parse_cropobject_list
import matplotlib.pyplot as plt

# Bear in mind that the outlinks are integers, only valid within the same document.
# Therefore, we define a function per-document, not per-dataset.


def extract_notes_from_doc(cropobjects):
    """Finds all ``(full-notehead, stem)`` pairs that form
    quarter or half notes. Returns two lists of CropObject tuples:
    one for quarter notes, one of half notes.

    :returns: quarter_notes, half_notes
    """
    _cropobj_dict = {c.objid: c for c in cropobjects}

    notes = []
    for c in cropobjects:
        if (c.clsname == "notehead-full") or (c.clsname == "notehead-empty"):
            _has_stem = False
            _has_beam_or_flag = False
            stem_obj = None
            for o in c.outlinks:
                _o_obj = _cropobj_dict[o]
                if _o_obj.clsname == "stem":
                    _has_stem = True
                    stem_obj = _o_obj
                elif _o_obj.clsname == "beam":
                    _has_beam_or_flag = True
                elif _o_obj.clsname.endswith("flag"):
                    _has_beam_or_flag = True
            if _has_stem and (not _has_beam_or_flag):
                # We also need to check against quarter-note chords.
                # Stems only have inlinks from noteheads, so checking
                # for multiple inlinks will do the trick.
                if len(stem_obj.inlinks) == 1:
                    notes.append((c, stem_obj))

    quarter_notes = [(n, s) for n, s in notes if n.clsname == "notehead-full"]
    half_notes = [(n, s) for n, s in notes if n.clsname == "notehead-empty"]
    return quarter_notes, half_notes

def get_image(cropobjects, margin=1):
    """Paste the cropobjects' mask onto a shared canvas.
    There will be a given margin of background on the edges."""

    # Get the bounding box into which all the objects fit
    top = min([c.top for c in cropobjects])
    left = min([c.left for c in cropobjects])
    bottom = max([c.bottom for c in cropobjects])
    right = max([c.right for c in cropobjects])

    # Create the canvas onto which the masks will be pasted
    height = bottom - top + 2 * margin
    width = right - left + 2 * margin
    canvas = np.zeros((height, width), dtype='uint8')

    for c in cropobjects:
        # Get coordinates of upper left corner of the CropObject
        # relative to the canvas
        _pt = c.top - top + margin
        _pl = c.left - left + margin
        # We have to add the mask, so as not to overwrite
        # previous nonzeros when symbol bounding boxes overlap.
        canvas[_pt:_pt+c.height, _pl:_pl+c.width] += c.mask

    canvas[canvas > 0] = 1
    print(canvas.shape)
    return canvas

def show_mask(mask):
    plt.imshow(mask, cmap='gray', interpolation='nearest')
    plt.show()

def show_masks(masks, desc, row_length=5):
    n_masks = len(masks)
    n_rows = n_masks // row_length + 1
    n_cols = min(n_masks, row_length)
    fig = plt.figure()
    for i, mask in enumerate(masks):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(mask, cmap='gray', interpolation='nearest')
    # Let's remove the axis labels, they clutter the image.
    for ax in fig.axes:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
    
    plt.savefig(f"notes{desc}.png")
    plt.show()

# Change this to reflect wherever your MUSCIMA++ data lives
CROPOBJECT_DIR = os.path.join("datasets/vision/muscima/v0.9.1/data/cropobjects")
cropobject_fnames = [
    os.path.join(CROPOBJECT_DIR, f) for f in os.listdir(CROPOBJECT_DIR)
]
docs = [parse_cropobject_list(f) for f in cropobject_fnames]

qns_and_hns = [extract_notes_from_doc(cropobjects) for cropobjects in docs]
qns = list(itertools.chain(*[qn for qn, hn in qns_and_hns]))
hns = list(itertools.chain(*[hn for qn, hn in qns_and_hns]))

print(len(qns), len(hns))

qn_images = [get_image(qn) for qn in qns]
hn_images = [get_image(hn) for hn in hns]

show_masks(qn_images[:25], "qn")
show_masks(hn_images[:25], "hn")