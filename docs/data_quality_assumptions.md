# Data Quality Assumptions

Assumptions derived from profiling the Amazon Fashion dataset (30k products).
These govern preprocessing, feature extraction, and similarity search behaviour.

---

## 1. Identifiers

- `uniq_id` has **zero duplicates** and is safe to use as the primary key.
- `asin` is Amazon's product identifier; not checked for cross-row duplicates but not used as a feature.

## 2. Sentinel & Missing Values

| Column | Assumption |
|---|---|
| `weight` | All 30k values are the sentinel `999999999`. Treat as NaN (100% missing). |
| `sales_price` | String in raw data; coerced to `float64`. ~9.6% missing (NaN after coercion). Zero values, if any, are treated as missing. |
| `rating` | String in raw data; coerced to `float64`. 100% non-null. All values within [1.0, 5.0]. |
| `discount_percentage` | Coerced to `float64`. 100% NaN after coercion — column is unusable. |
| `no__of_reviews` | ~11.5% coverage only. Too sparse for a reliable feature. |
| `left_in_stock` | ~10.2% coverage. Not used as a feature. |

## 3. Column Naming Mismatches

The feature specification uses names that differ from the dataset:

| Spec name | Actual column | Resolution |
|---|---|---|
| `color` | `colour` | British spelling; use `colour` everywhere in code |
| `price` | `sales_price` | No separate `price` column exists |
| `description` | `meta_keywords` | No `description` column; `meta_keywords` is the text proxy |
| `image_urls` | `image_urls__small` | Pipe-separated URLs; also available as `medium` and `large` |

## 4. Missing-Data Drop Thresholds

### Column-level (applied first)

- Columns with **>60% missing values** are dropped entirely rather than imputed — the signal-to-noise ratio is too low to be useful.
- Dropped columns: `weight` (100%), `discount_percentage` (100%), `left_in_stock` (~89.8%), `no__of_reviews` (~88.5%), `colour` (~79.9%).
- Note: `colour` is borderline but its combination of ~80% missing **and** very high cardinality (~4,757 unique) makes imputation unreliable. It is excluded from structured features.

### Row-level (applied second)

- Rows where **>50% of remaining columns are null** are dropped before feature extraction.
- Observed: **284 rows** exceed this threshold in the raw dataset.

## 5. Numeric Columns

- Raw values are strings; coerced via `pd.to_numeric(errors="coerce")`.
- **StandardScaler** is chosen over MinMaxScaler because `sales_price` has heavy outliers (max ≫ median), making min-max scaling compress the useful range.
- Missing numeric values are imputed with **column-wise median** (robust to skew).

## 6. Categorical Columns

| Column | Coverage | Cardinality | Strategy |
|---|---|---|---|
| `brand` | ~72.9% | ~6,458 | High cardinality → LabelEncoder. Missing → fill with `"unknown"`. |
| `colour` | ~20.1% | ~4,757 | Dropped (>60% missing + high cardinality makes imputation unreliable). |
| `delivery_type` | 100% | 2 | Low cardinality → one-hot or direct use. |
| `amazon_prime__y_or_n` | 100% | 2 | Binary flag. |
| `best_seller_tag__y_or_n` | 100% | 2 | Extremely imbalanced (99.96% = N). Low discriminative value. |

## 7. Text Columns

| Column | Coverage | Avg length | Assumption |
|---|---|---|---|
| `product_name` | 100% | ~58 chars | Primary text feature. Always present. |
| `meta_keywords` | 100% | ~78 chars | Proxy for `description`. Concatenated with `product_name` for embedding. |
| `other_items_customers_buy` | ~81.2% | ~500 chars | Not used for similarity (describes other products, not this one). |

- Empty text fields produce a **zero vector** as the embedding fallback.

## 8. Image URLs

| Column | Coverage | Avg URLs/product |
|---|---|---|
| `image_urls__small` | ~100% (29,998) | 4.2 |
| `medium` | ~100% (29,998) | 4.2 |
| `large` | ~96.1% (28,841) | 4.0 |

- URLs are **pipe-separated** (`|`). Only the **first URL** is used for feature extraction.
- Missing or unavailable images produce a **zero vector** as the embedding fallback.

## 9. Similarity Engine Defaults

| Parameter | Value | Rationale |
|---|---|---|
| Text weight | 0.40 | Highest-signal modality; always available |
| Image weight | 0.30 | Strong visual signal for fashion products |
| Structured weight | 0.30 | Price/rating/brand provide complementary signal |
| Fallback (no image) | text=0.60, structured=0.40 | Redistribute image weight proportionally |
| Tie-breaking | `sales_price` ascending | Cheaper products surface first on ties |

## 10. Crawl Metadata

- `crawl_timestamp` spans **2020-02-06 to 2020-02-07** (single crawl window).
- No temporal variation — time-based deduplication is unnecessary.
