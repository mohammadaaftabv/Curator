# Text pipeline language ID

`run_text_pipeline.py` supports the existing LLM language-ID stage plus local
FastText and IndicLID CPU backends. All backends write
`llm_language_prediction`; the existing verifier updates `_skipme`.

Both CPU backends use the `fasttext==0.9.3` runtime and a separate `.bin`
checkpoint. In an existing container, place `fasttext==0.9.3` and
`numpy==1.26.4` in an isolated Python overlay and prepend it to `PYTHONPATH`;
FastText 0.9.3 batch prediction is not compatible with NumPy 2.x. The LLM
backend does not use this overlay.

## One CPU backend

Use FastText when every input `source_lang` has an exact `lid.176` label:

```bash
python examples/audio/text_processing/run_text_pipeline.py \
  --input_manifest /data/input.jsonl \
  --output_dir /data/output \
  --enable_language_id \
  --language_id_backend fasttext \
  --fasttext_lid_model_path /models/lid.176.bin
```

To use IndicLID instead, replace the final two options with:

```bash
--language_id_backend indiclid \
--indiclid_lid_model_path /models/indiclid-ftn/model_baseline_roman.bin
```

## Route by language

The evaluated 22-language Granary routing is checked in as
`language_id_backends_indic_22.json`. It uses IndicLID for Assamese (`as`),
Bodo (`brx`), Dogri (`doi`), Gujarati (`gu`), Kashmiri (`ks`), Konkani
(`kok`), Maithili (`mai`), Malayalam (`ml`), Manipuri (`mni`), Nepali (`ne`),
Odia (`or`), Sanskrit (`sa`), Santali (`sat`), Urdu (`ur`), and Arabic-script
Sindhi (`sd`). It uses FastText for Bengali (`bn`), Hindi (`hi`), Kannada
(`kn`), Marathi (`mr`), Punjabi (`pa`), Tamil (`ta`), and Telugu (`te`):

```json
{
  "as": "indiclid",
  "bn": "fasttext",
  "brx": "indiclid",
  "doi": "indiclid",
  "gu": "indiclid",
  "hi": "fasttext",
  "kn": "fasttext",
  "kok": "indiclid",
  "ks": "indiclid",
  "mai": "indiclid",
  "ml": "indiclid",
  "mni": "indiclid",
  "mr": "fasttext",
  "ne": "indiclid",
  "or": "indiclid",
  "pa": "fasttext",
  "sa": "indiclid",
  "sat": "indiclid",
  "sd": "indiclid",
  "ta": "fasttext",
  "te": "fasttext",
  "ur": "indiclid"
}
```

Pass the checked-in config and both checkpoints:

```bash
python examples/audio/text_processing/run_text_pipeline.py \
  --input_manifest /data/input.jsonl \
  --output_dir /data/output \
  --enable_language_id \
  --language_id_backend config \
  --language_id_backend_config_file examples/audio/text_processing/language_id_backends_indic_22.json \
  --fasttext_lid_model_path /models/lid.176.bin \
  --indiclid_lid_model_path /models/indiclid-ftn/model_baseline_roman.bin
```

The recommended split is based on the evaluated per-language accuracy, not
only model label availability. It intentionally routes some languages that
have exact FastText labels to IndicLID and routes Konkani to IndicLID rather
than treating FastText's `gom` label as an exact `kok` match.

The `sd` route is specifically for Arabic-script Sindhi: IndicLID-FTN exposes
`snd_Arab` but no Devanagari Sindhi label. Routing currently uses only
`source_lang`, so a mixed-script `sd` cohort must be split or validated before
using this config. CPU backends read `abbreviated_text` before PnC. A config
must contain every `source_lang` present in the run; missing routes fail
closed. Omitting `--language_id_backend` preserves the existing LLM behavior.

Models: [FastText lid.176](https://fasttext.cc/docs/en/language-identification.html)
and [IndicLID-FTN v1.0](https://github.com/AI4Bharat/IndicLID/releases/tag/v1.0).
