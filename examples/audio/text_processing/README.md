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

Create a JSON file containing every `source_lang` present in the run:

```json
{
  "hi": "fasttext",
  "brx": "indiclid",
  "kok": "indiclid"
}
```

Then pass both checkpoints:

```bash
python examples/audio/text_processing/run_text_pipeline.py \
  --input_manifest /data/input.jsonl \
  --output_dir /data/output \
  --enable_language_id \
  --language_id_backend config \
  --language_id_backend_config_file /configs/lid_backends.json \
  --fasttext_lid_model_path /models/lid.176.bin \
  --indiclid_lid_model_path /models/indiclid-ftn/model_baseline_roman.bin
```

CPU backends read `abbreviated_text` before PnC. FastText fails closed for
languages without an exact model label; route those languages, including the
qualified Konkani `kok`/`gom` proxy, to IndicLID. Omitting
`--language_id_backend` preserves the existing LLM behavior.

Models: [FastText lid.176](https://fasttext.cc/docs/en/language-identification.html)
and [IndicLID-FTN v1.0](https://github.com/AI4Bharat/IndicLID/releases/tag/v1.0).
