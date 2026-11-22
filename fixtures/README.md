# Fixtures for the C++ and CUDA ports

Saved Python inputs and expected outputs. A port reads the inputs, writes its own outputs in the same format, and is
checked with:

```bash
python -m adi_heston.fixtures check fixtures/small path/to/port_output
```

The file format, the contents of each case and the tolerances are described in
[`docs/phase1_summary.md`, Section 4.5](../docs/phase1_summary.md#45-checking-a-port-against-python).

To regenerate these files (needed whenever a Python kernel changes):

```bash
python -m adi_heston.fixtures generate                 # tiny, small and the 7 variant cases (committed)
python -m adi_heston.fixtures generate --case medium   # not committed
```
