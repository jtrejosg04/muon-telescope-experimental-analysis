# Muon Telescope Experimental Analysis

Pipelines for the reconstruction and temporal analysis of the ground-level muon flux measured with two scintillator telescopes:

- **25×25 cm²** triple-coincidence telescope
- **60×60 cm²** double-coincidence telescope

The analysis includes geometric acceptance calculation, TDC/QDC quality cuts, live-time determination, flux reconstruction with the corresponding uncertainty budget, local significance analysis using moving medians (4 h, 6 h and 8 h), and robustness tests against window choice and binning.

## Structure

- `25x25/analisis_25x25.ipynb`
- `60x60/analisis_60x60.ipynb`

## Requirements

Python 3.8+ with `numpy`, `pandas`, `matplotlib`, `scipy` and `tqdm`.

## Usage

Open the corresponding notebook and set the data path at the beginning of the file. Both pipelines produce TDC/QDC histograms, native-resolution flux time series, and a candidate table when events pass the robustness criteria.

## Citation

Trejos-Giraldo, J. (2026). *Estudio del efecto de campos eléctricos atmosféricos en el flujo de muones a nivel del suelo*. Universidad de los Andes.

## License

MIT
