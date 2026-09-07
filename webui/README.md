# KARD Streamlit Interface

Human Activity Recognition — real-time dashboard.

## Structure

```
kard_streamlit/
├── app.py                  ← entry point
├── requirements.txt
├── pages/
│   ├── overview.py         ← project summary & progress
│   ├── inference.py        ← ★ LIVE INFERENCE (placeholders here)
│   ├── dataset.py          ← KARD dataset description
│   ├── pipeline.py         ← pre-processing & feature engineering
│   ├── models.py           ← model cards with params & results
│   └── results.py          ← comparison table & CV summary
└── README.md
```

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Linking inference.py  (the only thing you need to edit)

Open `pages/inference.py` and replace the two placeholders at the top:

```python
# PLACEHOLDER 1 — single video frame (numpy array H×W×3)
video_frame = None
# ↓ replace with:
from inference import get_current_frame
video_frame = get_current_frame()

# PLACEHOLDER 2 — confidence dict { activity_name: float }
class_confidences = { ... zeros ... }
# ↓ replace with:
from inference import get_confidences
class_confidences = get_confidences()
```

Enable **Auto-refresh** in the interface to poll the stream continuously.

## Notes

- `video_frame`      : `numpy.ndarray` shape (H, W, 3), BGR or RGB
- `class_confidences`: `dict` with exactly 11 keys (one per class), values in [0, 1]
- Confidence bars are sorted DESC automatically — no need to sort in inference.py
- The rest of the UI (all other pages) is fully independent of inference.py
