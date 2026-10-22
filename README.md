# Reading vs. Seeing

**Resolution, script, and text-priority bias in multi-scale typographic images**

## Overview

This project studies how frozen contrastive vision-language models interpret synthetic typographic images where a large word is filled with repeated smaller text. The large word (the **shape**) and the repeated text (the **fill**) can represent conflicting concepts. We test whether model predictions change with image resolution, whether the fill is written in Latin or Bengali script, and whether the fill contains meaningful words or meaningless strings.

The benchmark currently evaluates:

- **CLIP ViT-B/32** — `openai/clip-vit-base-patch32`
- **SigLIP 2 Base** — `google/siglip2-base-patch16-224`

The models remain frozen and are evaluated zero-shot using the same ten English candidate prompts. No fine-tuning is performed.

## Research questions

1. Does the model's preference shift from the large word to the small fill as resolution increases?
2. Does the fill begin to influence predictions at approximately the same resolution at which the fill word becomes independently legible?
3. Does the Latin-versus-Bengali fill script affect the results, and does this pattern differ between the two models?
4. Does meaningful text interfere with shape recognition more than random same-script strings?
5. Do transition patterns differ by model and script, making a single resolution threshold inappropriate?

## Experimental conditions

| Condition | Description | Role |
|---|---|---|
| `word` | Large shape concept S, different repeated fill concept F | Main conflict condition |
| `same` | Shape and fill refer to the same concept | Congruent control |
| `random` | Large shape with meaningless strings in the same script | Text-like texture control |
| `solid` | Large shape with a solid dark fill | No-small-text baseline |
| `tiled` | Repeated fill word across the image, without a large shape | Direct legibility probe |

The ten concepts are cat, dog, car, bus, hat, cow, pig, bee, fox, and egg. The large shape is always written in Latin letters. Fill text is Latin or Bengali.

## Main measurements

- **Tiled-word accuracy:** whether the model identifies the repeated fill word when it appears without a large shape.
- **Shape win / fill win:** whether the top-scoring English candidate matches the large word or the small fill.
- **Similarity margin:** cosine similarity for the shape concept minus cosine similarity for the fill concept. Positive values favour the shape; negative values favour the fill.
- **Crossover resolution (`R_star`):** first observed positive-to-nonpositive crossing of the mean similarity margin, interpolated on log resolution.
- **Legibility threshold (`R_legible_50pct`):** first resolution at which tiled-word accuracy reaches 50%.
- **Legibility/capture association:** exploratory Spearman correlation between independent legibility and fill capture.

These are operational definitions. A missing crossover is reported as no observed crossover in the tested resolution range; it should not be replaced by an invented threshold.

## Run the notebook

1. Open `reading_vs_seeing_rewritten.ipynb` in Google Colab or a Linux Jupyter environment.
2. Select a GPU runtime if available. The notebook can run on CPU, but inference will be slower.
3. Run the setup cell and **restart the runtime** as instructed.
4. Run cells from top to bottom.
5. **Inspect the generated images before model inference.** Verify that all Bengali words and signs are rendered correctly. Stop if shaping is broken.
6. The pilot chooses the smallest tested fill font size where CLIP reaches at least 80% accuracy on Latin tiled words at 224 px. If no size meets the rule, the largest tested size is used and a warning is printed.
7. Run the full experiment. The model downloads and inference stage can take time. Save `results/` to persistent storage before closing the runtime.

## Outputs

The notebook writes outputs into `results/`:

- `examples.png` — image-rendering and resolution sanity check
- `pilot_font_size.csv` — pilot accuracy by font size and script
- `manifest.csv` — item-level experimental design
- `similarities_clip-b32.npy` and `similarities_siglip2-base.npy` — similarity tensors
- `long_results.csv` — per-item, per-resolution predictions and outcomes
- `figA_legibility.png` — tiled-word legibility curves
- `figB_shape_vs_fill.png` — shape/fill predictions and similarity margins
- `summary_thresholds.csv` — crossover and legibility-threshold summaries
- `figD_legibility_gating.png` — legibility versus fill capture
- `figE_reading_vs_texture.png` — meaningful-text versus texture comparisons
- `high_resolution_condition_summary.csv` — high-resolution condition summary
- `run_info.json` — environment and experiment metadata

## Reproducibility

The notebook fixes random seeds and records model/library versions and the selected pilot font size. Exact results may still vary with hardware, library versions, and model implementation changes. Keep the output tables and figures with the notebook version used to produce them.

## Important limitations

- Only two frozen contrastive models are tested.
- There are ten benign concepts and one font family per script.
- The large shape is always Latin, and candidate prompts are always English.
- Synthetic typography does not represent all real-world text-in-image settings.
- Equal font sizes do not imply equal legibility across scripts.
- Random strings are not perfectly matched to real words in length or visual statistics.
- Observations can share concepts and rendering characteristics, so naive independence assumptions may understate uncertainty.
- Findings do not automatically generalise to generative VLMs, all Bengali text, real-world images, or safety outcomes.

## Responsible reporting

This notebook is an experimental scaffold, not a pre-written result. Report the measurements produced by your own completed run, include negative or inconclusive findings, document the Bengali rendering check, and avoid claiming that a hypothesis is supported unless the output actually supports it.

## Suggested extensions

- Evaluate a generative vision-language model on a subset.
- Add Bengali large-shape typography and cross shape script with fill script.
- Add another multilingual encoder.
- Improve uncertainty estimates by resampling at concept-pair or item-generation level and pre-registering analysis decisions.
