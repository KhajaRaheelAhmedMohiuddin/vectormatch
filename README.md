# VectorMatch v1.1.0 release assets

Binary artifacts kept off `main` to keep the code history small. Used by [`hub/publish.py`](https://github.com/KhajaRaheelAhmedMohiuddin/vectormatch/blob/main/hub/publish.py).

| File | Contents |
|---|---|
| `bge-small-vectormatch.zip.part0`, `.part1` | Fine-tuned embedder (split to stay under GitHub's 100 MB file limit; concatenate in order) |
| `vectormatch-space.zip` | Hugging Face Space bundle: Gradio app, engine, 25K jobs + fine-tuned embeddings |
| `SHA256SUMS` | Checksums of the reassembled zips |
