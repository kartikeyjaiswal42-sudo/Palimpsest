# Running the Binoculars notebook on Colab — step by step

You do not need to know Colab. It is a free Google service that runs Python notebooks on
Google's GPUs; you use it through a browser tab, logged in with a normal Google account.
Total hands-on time is about five minutes, plus ~20 minutes of waiting while it runs.

---

## Step 0 — build the bundle (on this machine, ~15 seconds)

```bash
cd palimpsest
python scripts/export_for_colab.py
```

This creates a `colab_bundle/` folder holding two files:

| file | what it is |
|---|---|
| `documents.jsonl` | 3,860 essays plus the exact sentence spans to score (11.9 MB) |
| `palimpsest_binoculars.py` | this repository's own scorer |

**Before you upload, know what you are uploading.** `documents.jsonl` contains real student
essays. Putting it on Colab sends that text to Google — the same fact the app reports as
`textLeavesMachine`. This is fine for research; it is just a decision worth making on
purpose rather than by accident. `python scripts/export_for_colab.py --no-text` builds an
inspectable copy with the essays stripped out if you want to see the shape first.

---

## Step 1 — put the two files in Google Drive

1. Open <https://drive.google.com>
2. Make a new folder in **My Drive** called exactly `palimpsest`
3. Drag both files from `colab_bundle/` into it

Drive is worth the extra minute. Colab hands out free GPUs and can take one back at any
time; with Drive the run **checkpoints there and resumes**, and you never re-upload 12 MB.
If you skip this the notebook falls back to an upload button, which also works — you will
just re-upload after any disconnect.

---

## Step 2 — open the notebook

1. Go to <https://colab.research.google.com>
2. **File → Upload notebook**
3. Choose `palimpsest/notebooks/binoculars_colab.ipynb`

---

## Step 3 — ask for a GPU

**Runtime → Change runtime type → T4 GPU → Save.**

This is the one step people forget. On CPU the run takes hours instead of minutes. The
notebook's first cell prints which GPU you were given and tells you which model pair fits.

---

## Step 4 — run it

**Runtime → Run all.**

Then watch for three things:

1. **A Drive permission popup.** Click through it — this is Colab asking to read the folder
   you made in step 1. It is expected.
2. **A sanity check, near the top.** It scores two short passages whose answer is already
   known and prints `ordering OK`. If it prints `WRONG`, stop: something about the model
   pair is broken and the full run would be wasted.
3. **A progress line** every 200 documents, with an ETA.

Roughly 20 minutes on a T4 with the default pair.

**If the session disconnects:** reconnect and Run all again. With Drive it resumes from the
last checkpoint and only scores what is left.

---

## Step 5 — read the summary before downloading

The second-to-last cell prints a median score per source, lowest (most machine-like) first.

**An earlier version of this guide said "if the sources are all mixed together, something
went wrong — do not download it." That advice was wrong and it has been removed.** When this
was actually run, the table *was* mixed, and the mixing was the result rather than a fault:

* older and cheaper generators (`liang_college_gpt3`, `machine_claude`, Gemini 3.1
  flash-lite) sat at the machine-like end, as predicted;
* **frontier models sat at the bottom — scoring as more human-like than the human sources**;
* **TOEFL, real second-language student writing, was the most machine-like source of all.**

That is the published Binoculars-on-frontier degradation and this project's own ESL
false-positive direction, visible in one table. Telling you to abort on seeing it would have
thrown away the finding.

So: read the table, do not judge the run by it. The only thing here that indicates a broken
run is the `ordering OK` sanity check in step 4, which uses two passages whose answer is
already known. The real measurement happens back in the repository.

---

## Step 6 — bring it home

The last cell downloads `binoculars_scores.jsonl` (or points you at it in Drive). Then:

```bash
# check the alignment WITHOUT writing anything
python scripts/join_binoculars.py --scores ~/Downloads/binoculars_scores.jsonl --dry-run

# if that passes, write it
python scripts/join_binoculars.py --scores ~/Downloads/binoculars_scores.jsonl

# measure the new column
python scripts/binoculars_probe.py
```

Always run `--dry-run` first. The join refuses on any misalignment rather than writing a
matrix that silently describes the wrong sentences, and the dry run shows you that verdict
before anything on disk changes.

---

## Which model pair?

Cell 3 offers three. Edit the one line `OBSERVER, PERFORMER = ...`:

| pair | needs | notes |
|---|---|---|
| `QWEN_0_6B` | ~2 GB | already proven to run; the weakest instrument |
| `QWEN_1_5B` | ~4 GB | **the default**; comfortable on a free T4 |
| `FALCON_7B` | ~15.5 GB | the Binoculars paper's own pair; needs Colab Pro (A100/L4) |

`binoculars.py` notes that published ablations show accuracy falling with pair size, so
`FALCON_7B` is the strongest version of this experiment. On a free 15 GB T4 it is too tight
to be safe.

**There is deliberately no 8-bit option.** An earlier draft had one and it was dead code —
the scorer takes no quantization parameter, so the flag did nothing. It was removed rather
than wired up: the scorer already forces every softmax to float32 because this statistic is a
quotient of two averaged logs where small errors do not cancel, and quantising the weights
underneath that is the same objection one level down. If Falcon does not fit, run a smaller
pair and **say which pair produced the number**.

---

## Things that go wrong, and what they mean

| what you see | what it is |
|---|---|
| `NO GPU` | Step 3 was skipped. Runtime → Change runtime type → T4. |
| `ordering WRONG` in the sanity check | The pair or the dtype is broken. Do not run the full scoring. |
| `still missing: [...]` | The two files are not in `MyDrive/palimpsest`, or the folder name differs. |
| `ValueError: ... do not share a vocabulary` | You mixed two model families. Both must be base/instruct of the *same* model. |
| `CUDA out of memory` | The pair is too big for this GPU. Pick a smaller one. |
| Join says `REFUSED` | Read the reason it prints — a missing document, an off-by-one in sentence indices, or a corrupt file. It names which. |
| N documents `FAILED` in the notebook | They are absent from the output, so the join will refuse that set. Re-run to retry them. |

---

## What this experiment can and cannot settle

Worth knowing before spending the time, because the honest expected outcome is modest.

`docs/11-build-and-limits.md` §6.5 rates this as *"worth running precisely because it is a
real experiment with a real chance of a negative result"*, and predicts it will *"move the
ceiling a little rather than break it"* — published Binoculars numbers also degrade on
frontier prose.

And whatever it shows, it inherits the corpus's limits: every machine essay here comes from
our own generation pipeline, and `docs/12` records a signal that scored 0.960 AUROC on that
pipeline and **0.490** on somebody else's. A good number here is a reason to run
`scripts/consensus_controls.py`, not a reason to believe it yet.
