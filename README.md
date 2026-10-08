# Flytype

A real fruit fly brain, simulated neuron by neuron, now learning to type on [Monkeytype](https://monkeytype.com)!

165,122 neurons. 10,228,000 signed synaptic connections. Every one of them
measured from an actual male *Drosophila melanogaster* by electron microscopy —
not invented, not sampled from a distribution, not a neural network "inspired
by" a brain.

## Demo

https://github.com/user-attachments/assets/dceae265-9131-4046-873b-e4c87d6f5279

## How it works

Flytype connects the visual processing and neural activity of a simulated fruit fly brain to a browser to read and type characters.

1. **Vision (The Retina):** A headless Playwright browser captures screenshots of the Monkeytype text. The active word is cropped into individual letter images (padded to 69px). These images are downsampled and fed into the fly's **892 retinotopic hex columns** (specifically L1 and L2 lamina monopolar cells).
2. **Brain Processing:** The visual signal propagates through the complete 165,122-neuron network. The simulation runs for 50 steps at a maximum rate of 180 Hz for each character, integrating the signal over time.
3. **Typing (The Output):** A trained logistic classifier reads the spike counts from a selected set of informative neurons in the brain, mapping the brain's internal representation to a specific lowercase character, which is then typed into the browser. 
4. **Learning:** Just like a real fly, the simulation utilizes dopamine-modulated plasticity at the Kenyon cell to Mushroom Body Output Neuron (MBON) synapses, subtracting connections based on a calculated reward signal.

## Running Flytype

Flytype's brain operates at ~100 WPM. To start the live dashboard locally:

```bash
# Run the local backend
python flytype_live.py
```
Then open `http://localhost:4652` in your browser. (Note: On Windows, use `localhost` or `127.0.0.1` instead of `0.0.0.0`).

### Benchmarks

To evaluate the accuracy of the brain's representation compared to raw pixels or basic L1 rates, we provide a validation benchmark:
```bash
python flytype_benchmark.py --audit-only
```
This tool validates how well the brain retains letter identity after simulating for 50, 100, or 200 steps.

## Installation

```bash
git clone https://github.com/fruitflydev/flycoinrh
cd flycoinrh
pip install -r requirements.txt
python -m playwright install chromium

# the connectome itself - 1.1 GB, CC-BY, no account and no key
# Download the feather files into data/ (see credits for source)
python build_graph.py            # -> build/graph.npz, 165,122 neurons
```

## Credits

Connectome data © HHMI Janelia FlyEM, the Cambridge Connectomics Group and
Google Research, released CC-BY. Simulation approach after Shiu et al. 2024 and
Lappalainen et al. 2024. Not affiliated with any of them.
MIT for the code. The connectome is **not ours to license** and stays CC-BY
wherever it goes — keep the attribution, it is the whole reason any of this is real.
