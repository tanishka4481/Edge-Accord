# Data Folder

Place the Speech Commands subset here when you are ready to train for real.

Suggested layout:

- `data/yes/*.wav`
- `data/no/*.wav`
- `data/stop/*.wav`
- `data/go/*.wav`
- `data/unknown/*.wav`
- `data/silence/*.wav`

The host-side training script can also run with toy data for smoke tests using `--toy-data`.
