# Feature Extraction Validation

Image encoder: `openai/clip-vit-base-patch32`, 512 dimensions, 9,479 of 9,503 items encoded.

Text encoder: `sentence-transformers/all-mpnet-base-v2`, 768 dimensions, 9,503 of 9,503 items encoded.

## Numerical integrity

All embeddings are finite. Embedding rows are aligned to `item_index.parquet` by position, verified by comparing the stored identifier order against the index.

## Embedding scale

Image vectors have mean norm 10.331; text vectors have mean norm 1.000, because the text encoder applies L2 normalisation internally. The resulting scale ratio of 10.3:1 requires per-modality normalisation before fusion, so that neither modality dominates the concatenated representation for reasons of numerical magnitude rather than information content.

## Semantic structure

Within-domain and cross-domain cosine similarity, averaged over 2,000 sampled pairs per domain:

| modality | domain | within | across | margin |
|---|---|---|---|---|
| image | laptops_tablets | 0.707 | 0.618 | +0.090 |
| image | audio | 0.720 | 0.632 | +0.089 |
| image | wearables | 0.776 | 0.600 | +0.176 |
| image | computer_accessories | 0.691 | 0.636 | +0.055 |
| image | camera_photo | 0.715 | 0.615 | +0.100 |
| text | laptops_tablets | 0.501 | 0.227 | +0.275 |
| text | audio | 0.468 | 0.199 | +0.270 |
| text | wearables | 0.612 | 0.235 | +0.376 |
| text | computer_accessories | 0.378 | 0.226 | +0.152 |
| text | camera_photo | 0.384 | 0.170 | +0.214 |

A positive margin in every domain confirms that both encoders place products of the same kind closer together than products of different kinds, which is the property the multimodal design depends on.
