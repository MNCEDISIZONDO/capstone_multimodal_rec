# Data Filtering Report

**Project:** Multimodal Recommender System for E-Commerce Item Cold-Start
**Dataset:** Amazon Reviews 2023, Electronics category
**Split construction seed:** 42 (frozen)

This report records every filtering stage applied to the raw data, the number of
users, items and interactions removed at each stage, and the reasoning behind
each decision. It is the source material for the methodology chapter.

---

## 1. Data sources

Two files were retrieved from the Amazon Reviews 2023 benchmark.

| File | Size | Records |
|---|---|---|
| `Electronics.csv.gz` (0-core, rating-only) | 925 MB | 43,365,426 interactions |
| `meta_Electronics.jsonl.gz` | 1.22 GB | 1,610,012 items |

**Rating-only interaction source.** The interaction file used is the rating-only
export, containing exactly four fields: user identifier, product identifier,
rating and timestamp. This project derives textual embeddings from product
metadata — titles, feature bullets and descriptions — and never from review
text, so the full review export would have contributed tens of gigabytes of
unused data.

**0-core rather than 5-core.** The benchmark publishes both an unfiltered
(0-core) and a pre-filtered (5-core) release. The 0-core release was selected
because the 5-core release removes all users and items with fewer than five
interactions before the data is seen, which would have made any independent
threshold analysis meaningless: the observed distribution would reflect the
publishers' filtering decisions rather than the underlying data.

**Product identifier.** The rating-only export is keyed on `parent_asin`, the
identifier that unifies product variants such as colour and storage capacity.
Joining on the variant-level identifier would fragment each product family
across many sparsely-observed records. Verified downstream: the converged corpus
shows a median of 16 interactions per item, consistent with a correct join.

**Documentation discrepancies encountered.** Two published details did not match
the retrieved artifacts and are recorded here for reproducibility. First, the
metadata URL listed on the dataset's public card was unreachable; the live host
is `mcauleylab.ucsd.edu`. Second, the `images` field is documented as an object
with named size keys, but the raw export uses a list of per-image objects, and
the high-resolution key is frequently null while the large key is populated.

---

## 2. Domain assignment

Items were assigned to five structurally adjacent sub-domains: laptops and
tablets, audio, wearables, computer accessories, and camera and photography.

The `categories` field is a hierarchical path running from broad to specific
rather than a flat set of tags. This property governs the entire assignment
design. A three-stage filter was applied.

**Stage 1 — leaf exclusion.** Accessory and component markers are matched
against the final path token only, because the final token identifies what the
item is. A laptop's path terminates in `traditional laptops`; a laptop bag's
terminates in `bags, cases & sleeves`. Both share the ancestor
`computers & accessories`.

> An initial implementation matched exclusion terms against every token in the
> path. Because the ancestor `computers & accessories` contains an exclusion
> term, this rejected every laptop, keyboard, monitor and pair of headphones in
> the catalogue, producing a 2,536:1 domain imbalance and only 14 surviving
> laptops. The failure is recorded because it is silent: the pipeline reported
> success and produced a plausible-looking corpus.

**Stage 2 — title exclusion.** The source data files straps, cases and chargers
under the category leaf of the device they accompany, so category rules alone
cannot separate them. Exclusion phrases were kept deliberately narrow to avoid
rejecting legitimate products.

**Stage 3 — title affirmation.** Each item's title must contain at least one
term relevant to its assigned domain. This stage handles outright
misclassification in the source metadata — instances observed include a
toothbrush and a towel filed under loudspeakers — which no category rule can
anticipate but which fail an affirmative title test immediately.

Over-rejection was treated as acceptable throughout, because the surviving pool
exceeds the corpus requirement by more than an order of magnitude, whereas
under-rejection contaminates the corpus irreversibly.

### Result

| | items |
|---|---|
| raw metadata | 1,610,012 |
| assigned to a domain | **190,705** |
| removed | 1,419,307 |

| domain | items |
|---|---|
| audio | 75,854 |
| computer_accessories | 39,356 |
| laptops_tablets | 37,278 |
| camera_photo | 31,242 |
| wearables | 6,975 |

Content availability across in-scope items: title 100%, image URL 100%,
feature bullets 82.4%, description 65.2%.

A manual audit of 25 randomly sampled titles per domain was performed after each
revision of the rules, using the frozen construction seed for reproducibility.

### Scope decisions

- **Desktops** are excluded from laptops_tablets; the domain covers portable
  computing.
- **Surveillance and vehicle cameras** are excluded from camera_photo; these are
  security and automotive products rather than consumer photography.
- **Tripods and monopods** are retained in camera_photo despite the general
  exclusion of accessories. The exclusion targets low-signal items such as
  cables and cases, which carry boilerplate text and packaging photography.
  Tripods are substantial products with distinctive visual and textual content.
- **Webcams** appear in camera_photo where their titles describe them as
  cameras. This residual is documented rather than corrected, as further rules
  would risk excluding legitimate products.
- No item matched more than one domain, since the inclusion vocabulary uses
  mutually exclusive device-level category tokens.

Residual misclassification is non-zero and is reported as a limitation.
Subsequent density filtering removes much of it implicitly, since
misclassified items rarely accumulate sufficient interactions from
cross-domain purchasers.

---

## 3. Interaction filtering

Interactions were restricted to in-scope items.

| | interactions |
|---|---|
| raw | 43,365,426 |
| involving an in-scope item | **8,093,805** (18.7%) |

The retention rate exceeds the proportion of items retained (11.8%), indicating
that devices attract more interactions per item than the accessories and
consumables removed during domain assignment.

---

## 4. Feasibility census

Before committing to any threshold, the population of cross-domain purchasers
was measured directly, because the viability of the entire design depends on
their existence in sufficient number.

### Users by number of domains touched

| domains | users | share |
|---|---|---|
| 1 | 5,072,208 | 87.47% |
| 2 | 615,554 | 10.61% |
| 3 | 95,680 | 1.65% |
| 4 | 14,161 | 0.24% |
| 5 | 1,429 | 0.02% |

**111,270 users satisfy the three-domain constraint.** The constraint was
therefore retained at three domains, and the contingency of relaxing it to two
was not required.

### Median interactions per user, by domains touched

| domains touched | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| median interactions | 1.0 | 2.0 | 4.0 | 7.0 | 14.0 |

This monotonic relationship is the empirical justification for the cross-domain
constraint: it demonstrates that the filter selects for interaction density
rather than merely reducing the size of the dataset.

### Threshold sensitivity

Convergence was run at four candidate item thresholds to inform the choice.

| min. interactions per item | users | items | interactions | density |
|---|---|---|---|---|
| 3 | 38,846 | 25,149 | 327,697 | 0.0335% |
| 5 | 29,986 | 14,833 | 251,633 | 0.0566% |
| **7** | **23,310** | **9,503** | **191,763** | **0.0866%** |
| 10 | 13,646 | 3,995 | 104,545 | 0.1918% |

A threshold of ten produced the highest density but only 3,995 items, below the
corpus target, and reduced the wearables domain to 102 items. A threshold of
seven was selected as the setting that satisfies the corpus size requirement
while retaining adequate density in every domain.

---

## 5. Iterative filtering to convergence

The cross-domain user constraint and the minimum-interactions-per-item
constraint are mutually dependent: removing users starves items of interactions,
and removing items disqualifies further users. Applying each filter once leaves
the data in a state where neither condition holds. The two filters were
therefore alternated until a complete pass produced no further removals.

Applied thresholds: at least 3 domains per user, at least 5 interactions per
user, at least 7 interactions per item.

| | users | items | interactions | density |
|---|---|---|---|---|
| before | 5,799,032 | 190,688 | 8,093,805 | 0.0007% |
| after convergence | **23,310** | **9,503** | **191,763** | **0.0866%** |

Interaction density increased by a factor of 124. All three constraints were
verified to hold on the converged data by assertion.

### Corpus sizing

The converged corpus was adopted in full rather than sampled down to a smaller
target. A trial reduction toward 6,000 items produced a cascade collapse to
2,308 items: removing items disqualified their purchasers under the cross-domain
constraint, which starved further items in turn. Retaining the full converged
graph preserves corpus size, interaction density and the complete popularity
spectrum simultaneously.

Domain representation is therefore proportional to observed purchasing
behaviour rather than artificially balanced. Wearables is a structurally small
domain among cross-domain purchasers, a property of the data reported
transparently rather than engineered away. Notably, wearables exhibits the
highest interaction density of any domain (0.471%), indicating that its small
item count does not imply weak representations.

---

## 6. Image retrieval and the content floor

Product images were retrieved for all 9,503 corpus items before the cold-start
split was constructed, since cold-start eligibility requires a verified local
image. The large size variant was preferred: the vision encoder consumes images
at 224×224 pixels, so higher resolutions provide no benefit, and the
high-resolution field is frequently absent.

Each download was retried with increasing delays and verified to be a decodable
image of sensible size, rather than an error page or placeholder.

| outcome | items | share |
|---|---|---|
| retrieved and verified | 9,479 | 99.7% |
| below minimum size | 23 | 0.2% |
| retrieval failed | 1 | 0.0% |

The retrieval rate substantially exceeds the 70–85% typically expected for this
dataset. The corpus consists of items with at least seven interactions from
active purchasers, which are established products whose listings remain
maintained; expired links concentrate among obscure items removed during earlier
filtering.

**Content floor.** Items possessing neither a usable image nor usable text were
to be removed, as such items are invisible to the content pathway of the model
and would be unrecommendable by any means in a cold-start setting. **No items
required removal**: all 9,503 items have a title, and 9,479 also have a verified
image. Twenty-four items are text-only and will require modality masking during
feature extraction rather than consumption of placeholder image vectors.

---

## 7. Cold-start split construction

Cold-start items are withheld entirely from training. The model observes no
interaction involving them and must recommend them from image and text alone,
simulating a newly listed product.

The order of operations was: finalise the corpus, select the cold-start items,
remove every interaction involving them, and only then partition the remainder.
Selecting cold-start items after partitioning would permit the model to learn
representations for them, converting the cold-start evaluation into a
warm-start evaluation reporting better results than the truth.

Selection was stratified by domain and by popularity band, restricted to items
with a verified image, usable text, and at least five interactions.

| | value |
|---|---|
| cold-start items | 1,901 (20.0%) |
| cold-start interactions | 39,774 |
| interactions remaining for training | 151,989 |
| orphaned users removed | 4 |

### Validation

| check | result |
|---|---|
| cold-start items appearing in training | 0 |
| cold-start items with an evaluable interaction | 1,901 / 1,901 |
| evaluated users absent from training | 0 |
| interactions appearing in more than one split | 0 |

### Representativeness

| domain | cold-start share | corpus share |
|---|---|---|
| laptops_tablets | 10.8% | 10.8% |
| audio | 39.1% | 39.1% |
| wearables | 4.0% | 4.0% |
| computer_accessories | 29.1% | 29.1% |
| camera_photo | 16.9% | 16.9% |

Mean interactions: 20.9 (cold-start) against 20.0 (warm). Median interactions:
12 against 12. The mean difference of 4.5% arises from the requirement that
cold-start items carry at least five interactions to be evaluable, which biases
selection marginally toward more established products. The identical medians
indicate the effect is confined to a small number of outliers.

---

## 8. Training, validation and test partition

The remaining interactions were partitioned by a chronological leave-one-out
scheme: each user's most recent interaction forms their test item, the second
most recent their validation item, and all earlier interactions form training.
Time-based partitioning reflects the deployed task of predicting a user's next
action from their history; random partitioning would permit prediction of
earlier behaviour from later behaviour.

Ties in timestamp are broken deterministically by product identifier. The source
data batch-records reviews, so identical timestamps occur frequently, and an
undefined tie-break would render the partition irreproducible.

Users with fewer than five remaining interactions were removed before
partitioning, so that at least three interactions remain in training after one
is allocated to validation and one to test.

| | users | interactions |
|---|---|---|
| before minimum-interaction filter | 23,306 | 151,989 |
| after filter | **15,794** | 125,289 |

| split | interactions | users | items |
|---|---|---|---|
| train | 93,701 | 15,794 | 7,551 |
| validation | 15,794 | 15,794 | 5,113 |
| test | 15,794 | 15,794 | 4,663 |
| cold-start evaluation | 25,656 | — | 1,901 |

**14,097 cold-start interactions were removed** because their users no longer
appear in training. Evaluating them would have measured user cold-start
performance, which this project does not address. All 1,901 cold-start items
retain at least one evaluable interaction.

**51 items appear in validation or test without any training interaction.**
These constitute accidental cold-start cases: the model holds no learned
representation for them. They represent 0.7% of items and are recorded for
exclusion during evaluation rather than corrected by reconstruction.

---

## 9. Dataset characterisation

### Popularity distribution

| measure | value |
|---|---|
| Gini coefficient | 0.463 |
| share of interactions held by top 1% of items | 13.1% |
| share held by top 5% | 30.2% |
| share held by top 10% | 41.1% |

The distribution is skewed, as is expected of e-commerce interaction data, but
not dominated by a small number of products. Conclusions drawn from this corpus
are therefore not confined to the characteristics of a few high-traffic items.

### Interaction density by domain

| domain | items | training interactions | density |
|---|---|---|---|
| wearables | 380 | 2,839 | 0.4710% |
| laptops_tablets | 1,025 | 9,357 | 0.1771% |
| camera_photo | 1,610 | 16,426 | 0.1569% |
| computer_accessories | 2,768 | 29,537 | 0.1096% |
| audio | 3,720 | 35,542 | 0.0934% |

Density is inversely related to domain size: the smallest domain is the densest
and the largest the sparsest. Domain size and representation quality are
therefore distinct properties, and per-domain performance differences should be
interpreted against density rather than attributed solely to model architecture.

---

## 10. Split freeze

The partition is frozen. SHA-256 checksums of each split file are recorded in
`reports/split_checksums.json` together with the construction seed. Every model,
every training seed and every evaluation runs against this partition.

Any rebuild of the split voids all results produced before it, which is why
regeneration must be explicit and recorded rather than incidental.

---

## 11. Summary

| stage | items | interactions |
|---|---|---|
| raw | 1,610,012 | 43,365,426 |
| after domain assignment | 190,705 | 8,093,805 |
| after convergence | 9,503 | 191,763 |
| after content floor | 9,503 | 191,763 |
| training pool after cold-start removal | 7,602 | 151,989 |
| final training partition | 7,551 | 93,701 |

Final corpus: 9,503 items across five domains, 15,794 users, 0.0866% corpus
density, 99.7% image availability, 1,901 cold-start items with verified zero
leakage.
