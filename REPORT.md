# CS4243 Lab 1 reference implementation report

This compact report records how every `YOUR CODE HERE` section is exercised. This is where you elaborate and explain your implementation and results you observed. Try to add visual/graphs and grounded reasoning to highlight your understanding.

## Gabor Implementation

`make_gabor_bank` validates that kernel_size is an odd integer of at least 3, constructs a 2D spatial coordinate grid centered at the origin, and loops over every combination of frequency, orientation, and phase to generate Gabor filters by multiplying a rotated sinusoidal carrier wave with a 2D Gaussian envelope. Each kernel is then made zero-mean to eliminate background bias, normalized to unit Euclidean norm, and paired with a metadata dictionary containing its parameters and full configuration in a list. [AI-CODE][AI-LANGUAGE][HUMAN-CHECK]

`gabor_energy_maps` takes these kernels and correlates each one with a floating-point grayscale image using correlate2d to extract directional edge and texture responses without altering spatial dimensions. It converts the signed responses into positive energy representations using either squared or absolute values, applies a pool_size * pool_size box mean filter to smooth out high-frequency spatial oscillations into continuous texture envelopes, stacks the K channels into an H * W * K float32 array, and returns the result after verifying that all values are finite.
[AI-CODE][AI-LANGUAGE][HUMAN-CHECK]

## Edge Implementation

`bilinear_sample` interpolates gradient magnitudes with fractional offsets based on the distance from the original coordinates and clamps boundary coordinates.[AI-LANGUAGE][HUMAN-CHECK]

`nms_interpolated` compares each magnitude value with samples one step forward and backward in the gradient direction adn retains only the local maxima.

`adaptive_thresholds` filters out non-maximum suppressed (nms) values that are not positive. If there are no values left, return positive infinity for both threshold values such that no strong/weak edges are produced. With the positive nms values, calculates high and low threshold values based on the threshold method (percentile or robust) and returns the calculated threshold values.[AI-LANGUAGE][HUMAN-CHECK]

`hysteresis` takes in strong and weak values that have been filtered by the high and low threshold values respectively. The connectivity values then determines what counts as "connected edges", either the 4 direct pixels surrounding or the 8 pixels surrounding. Then the function checks which weak edges are connected to strong edges, adding these connected edges to detected edges.[AI-LANGUAGE][HUMAN-CHECK]

`detect_edges` is the main orchestrator function, bringing the edge detection functions together with one simple operation in the middle where it filters for strong pixels and weak pixels.

## Features and Representations

`extract_local_features` standardises the input image represetnation by converting to float32, then turning grayscale image into 3 colour channels, then scaling values to 0 to 255 inclusive when necessary. Then it converts the image to grayscale once for features that operate using intensity. Based on the configurations, the functio includes colour channels, Gabor texture responses, gradient energy, edge density, and orientation specific edges into a single feature map. Then gradient energy is computed from the squared eghe magnitude and locally averaged, while edge orienation features dividfe unsigned gradient directions into evenly spaced bins across 0 to pi, and measure the local density within each orietation. Lastly, each channel is assigned a descriptive name and optional per image standardization normalises each feature channel using its mean and standard deviation. [AI-CODE][AI-LANGUAGE][HUMAN-CHECK]

`global_pool` converts the feature map into a flattened array while keeping the feature channel order. Then it computes the summary statisic for each channel, including mean, standard deviation, percentile, etc. according to the statistics required. The statistics vector are then concatenated into the final one dimenionsal descriptor. [AI-CODE][AI-LANGUAGE][HUMAN-CHECK]

`feature_family_indices` maps each feature channel to their broaded feature families.[AI-CODE][HUMAN-CHECK]

## Normality Model

The normality component is a per-material one-class anomaly detector. It learns the distribution of local handcrafted features from normal training images only; it does not fit on defective images. A separate diagonal normal reference model is fitted for each material, so a local feature is anomalous when it differs substantially from the normal feature values for that material.[AI-LANGUAGE][HUMAN-CHECK]

`fit_normal_model` validates that the input is a non-empty collection of three-dimensional `H × W × D` feature maps with a common feature dimension. It flattens every spatial feature map into patch vectors of length `D`, concatenates these vectors across normal training images, and reproducibly subsamples them when their number exceeds `max_samples`. It then calculates the per-feature mean and population standard deviation, forming a diagonal normal model rather than estimating correlations between feature channels. Every original normal map is scored by its root-mean-square standardized distance from this reference, spatially smoothed with a box mean, and cropped at the unreliable border. The function uses the 99.5th percentile of these normal-only scores as a provisional threshold unless a threshold is supplied explicitly, then stores the mean, standard deviation, threshold, and optional feature names in a `NormalModel`. The small `epsilon` in the denominator prevents a nearly constant feature channel from causing division by zero. [AI-LANGUAGE][HUMAN-CHECK]

`predict_anomaly` first verifies that a new image feature map has the same final feature dimension as the fitted model. At every spatial location, it calculates the root-mean-square of the per-channel z-scores, producing a dense anomaly map in which larger values represent features that are less typical of the material's normal training distribution. A local box mean suppresses isolated noisy responses and retains more coherent anomalous regions. The image-level anomaly score is the 99th percentile of the interior score map, while the binary predicted mask is produced by thresholding the full smoothed map. The configured five-pixel outer border is set to `False` in the mask because filtering near the image boundary is less reliable. The function returns the float32 score map, the scalar image-level score, and the Boolean predicted mask. [AI-LANGUAGE][HUMAN-CHECK]

`select_mask_threshold` calibrates the final pixel threshold on validation data only. It checks that the validation score maps and masks are non-empty, paired, shape-compatible, and finite, then flattens all validation pixels. It evaluates 80 candidate thresholds spanning score quantiles from 0.5 to 0.999. For each candidate, it predicts pixels with `score ≥ threshold`, calculates true positives, false positives, and false negatives, and selects the threshold with the greatest pixel-level F1 score. Keeping the first threshold when F1 values tie makes selection deterministic. This validation-selected threshold replaces the provisional normal-only threshold for every material model before held-out test evaluation, preventing test-set information from influencing threshold selection. [AI-LANGUAGE][HUMAN-CHECK]

Overall, the Task B workflow fits the five material-specific normal models on normal training images, generates validation score maps, removes the same unreliable border from validation masks and scores, selects one F1-optimal threshold, freezes this choice, and finally evaluates image-level detection and pixel-level segmentation on the held-out test split. [AI-LANGUAGE][HUMAN-CHECK]

## Results and Analysis Task A-C
Q3. 
![Channel Scale](./student_files/q3_channel_scale.png)
From the channel scale plot we can seee that the differenet families have very different numerical ranges. For example, the RGB colour channels have the largest mean values, while that of the Gabor and gradient energy channels are much smaller. [AI-LANGUAGE][HUMAN-CHECK]

The edge features also vary alot in standard deviation, where edge density have a higher standard deviation compared to the edge orientation channels. [AI-LANGUAGE][HUMAN-CHECK]

This means that in their raw form, some features will contribute more strongly to a linear classifier because their values are much larger. Therefore, scaling is important before fitting the data and values for logistic regression. [AI-LANGUAGE][HUMAN-CHECK]

StandardScaler centrers each dimension and scales it to unit varince, ensuring that the classifier learns from each feature rather than being biased to those with larger raw vlaues. It also makes regulatization in logistic regression behave more consistently across all feature dimensions. [AI-LANGUAGE][HUMAN-CHECK]

Across the entire MVTec training images, it achieved 94.95% accuracy and macro F1 score of 0.949. [AI-CODE][AI-LANGUAGE][HUMAN-CHECK]

<img src="./student_files/q3_confusion_matrix.png" width="300" alt="Confusion Matrix" />
<img src="./student_files/q3_conf_histogram.png" width="360" alt="Confidence Histogram" />

Carpet: Precision: 0.870, Recall: 1.000, F1-Score: 0.930

Grid: Precision: 1.000, Recall: 0.800, F1-Score: 0.889

Leather: Precision: 1.000, Recall: 0.950, F1-Score: 0.974

Tile: Precision: 0.909, Recall: 1.000, F1-Score: 0.952

Wood: Precision: 1.000, Recall: 1.000, F1-Score: 1.000

[AI-CODE][AI-LANGUAGE][HUMAN-CHECK]

Most predictions lie on the diagonal, which shows that the classes were generally well separated. The main source of confusion came from the second class, where 2 samples were predicted as class 1 and another 2 as class 4. There was also one sample from class 3 that was predicted as class 1. The other two classes were classified perfectly in this test set. [AI-LANGUAGE][HUMAN-CHECK]

The confidence histogram shows that the classifier is generally very confident when it is correct. Most correct predictions are concentrated in the 0.9–1.0 confidence range, which is consistent with the high overall accuracy. [AI-LANGUAGE][HUMAN-CHECK]

The incorrect predictions are much fewer and mostly appear at moderate to high confidence, around 0.5–0.6 and close to 0.9–1.0. This suggests that although most mistakes are relatively uncertain, there are still a few cases where the classifier is confidently wrong. [AI-LANGUAGE][HUMAN-CHECK]

![Error 1](./student_files/q3_error_1.png)
The first error was a grid predicted as carpet. The gradient-energy and edge-density maps show similar structural responses to the predicted class, while the Gabor response indicates a strong repeated texture pattern. Because the classifier uses globally pooled descriptors, these similar global texture statistics may have caused the confusion. [AI-LANGUAGE][HUMAN-CHECK]

![Error 2](./student_files/q3_error_2.png)
The second misclassification is a grid predicted as tile. Here, it shows similar edge and texture responses between the true and predicted classes. In particular, the dominant Gabor response is strong across much of the image, while the edge-density map contains a comparable overall structure. This suggests that the pooled handcrafted representation captures similar global statistics for the two classes, even though the original images are visually different. [AI-LANGUAGE][HUMAN-CHECK]

Q4. ![Confident Error](./student_files/q4_confident_error.png)
The classifier is less accurate on defective images because defects alter the local texture patterns that the model relies on for classification. From the diagram, the source image contains a visible defect that disrupts the normal surface structure. This is also reflected in the response maps, where the gradient-energy and edge-density maps show strong responses around the defective region, while the Gabor response highlights changes in the usual texture orientation and frequency. The abnormal responses shown can then make the defective image resemble another material when examining the features, causing the classifier to make a confident yet incorrect prediction.

### Task B
| Defect | F1 Score  (Gabor & Edge)| IoU (Gabor & Edge)| F1 Score (Gabor Only)| IoU (Gabor Only)| F1 Score (Edge Only)| IoU (Edge Only)|
|---|---|---|---|---|---|---|
| Colour | 0.307 | 0.181 | 0.296 | 0.173 | 0.384 | 0.238 |
| Cut | 0.0635 | 0.0328 | 0.0622 | 0.0321 | 0.0700 | 0.0363 |
| Good | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Hole | 0.134 | 0.0719 | 0.124 | 0.0660 | 0.138 | 0.0743 |
| Metal contamination | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Thread | 0.0868 | 0.0454 | 0.0794 | 0.0413 | 0.0801 | 0.0417 |
| Bent | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Broken | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Glue | 0.196 | 0.109 | 0.196 | 0.108 | 0.209 | 0.117 |
| Fold | 0.207 | 0.115 | 0.202 | 0.113 | 0.235 | 0.133 |
| Poke | 0.0275 | 0.0139 | 0.0270 | 0.0137 | 0.0333 | 0.0170 |
| Crack | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Glue strip | 0.00241 | 0.00120 | 0.00480 | 0.00241 | 0.0 | 0.0 |
| Gray stroke | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Oil | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Rough | 0.0175 | 0.00884 | 0.0591 | 0.0305 | 0.0 | 0.0 |
| Combined | 0.314 | 0.186 | 0.305 | 0.180 | 0.373 | 0.229 |
| Liquid | 0.530 | 0.360 | 0.572 | 0.401 | 0.425 | 0.270 |
| Scratch | 0.583 | 0.412 | 0.581 | 0.410 | 0.437 | 0.279 |
| Pixel| 0.168 | 0.0919 | 0.167 | 0.0909 | 0.162 | 0.0882 |

From the IoU and F1 Scores, we can tell that the defects "Colour", "Cut", "Hole", "Thread", "Glue", "Fold", "Poke" and "Combined" mainly create connected edges due to the higher IoU and Fi score using an edge only configuration. While defects like "Glue Strip", "Rough", "Liquid", "Scratch" and "Pixel" appear mainly as colour or diffuse texture changes.

Additionally, defects "Thread", "Scratch" and "Pixel" have higher IoU and F1 scores for the combined (Gabor & Edge) configuration. This likely indicates that colour or texture changes and connected edges are complementary for these defects.

Defect types with zero IoU and F1 score for all three configurations may have been caused by weak anomaly response or a small defect area.

Q5. Bright illumination near an image border can create strong gradients and unusual colour/Gabor responses. The model may mark the border as anomalous even when there is no defect which produces false-positive mask regions and an inflated image-level anomaly score. Small registration shifts can cause a similar problem where normal texture edges no longer align with those seen in training.[AI-LANGUAGE][HUMAN-CHECK]

If the aggregation percentile increases, the number of pixels classified as anomalous would either be the same or be less, hence the image score would either remain the same or increase.

If mask threshold were increased, fewer pixels would exceed the threshold, so predicted mask shrinks. This may lead to an increaase in False Negatives (FN) or a decrease in False Positives (FP) or both, which may affect both F1 Score and IoU score, depending on the proportion of FN and FP change. Image score is not affected.

### Task C
| Attribute | Average Precision (AP) | F1 Score | Validation positives |
|---|---|---|---|
| Banded | 0.428 | 0.500 | 41 |
| Blotchy | 0.191 | 0.0 | 70 |
| Braided | 0.142 | 0.0 | 42 |
| Bumpy | 0.160 | 0.043 | 43 |
| Cracked | 0.122 | 0.0 | 40 |
| Fibrous | 0.132 | 0.0 | 43 |
| Grid | 0.249 | 0.233 | 41 |
| Marbled | 0.206 | 0.045 | 43 |
| Pitted | 0.081 | 0.0 | 43 |
| Porous | 0.109 | 0.0 | 45 |
| Stained | 0.177 | 0.067 | 41 |
| Striped | 0.555 | 0.554 | 51 |
| Woven | 0.332 | 0.179 | 46 |

Macro Average Precision and Macro F1 were 0.222 and 0.125 respectively. As the values for Average Precision and F1 are low to moderate values, it can be inferred that the handcrafted descriptor does not perfectly separate each texture. Comparing the scores, more structured textures such as "Banded" and "Striped" attained higher score due to the frequency and uniformity of the repeated lines.[AI-LANGUAGE][HUMAN-CHECK]

<img src="./student_files/NB3_TaskC_banded.png" width="160" alt="Banded Texture" />
<img src="./student_files/NB3_TaskC_striped.png" width="320" alt="Striped Texture" />

#### <u>Probability Heamap and Qualitative Example Analysis</u>

<img src="./student_files/NB3_TaskC_heatmap.png" height="350" width="700" alt="Predicted Probability Heatmap" />

<img src="./student_files/NB3_TaskC_banded.png" width="160" alt="Banded Texture" />
<img src="./student_files/NB3_TaskC_grid.png" width="160" alt="Grid Texture" />
<img src="./student_files/NB3_TaskC_stained.png" width="160" alt="Stained Texture" />

The grid example shows a regular, repeated lattice pattern with strong horizontal and vertical structure. This has orientation and spatial-frequency evidence and hence it would be well suited for Gabor filters. However, it can also be seen that the flaw in the Gabor filter is that it picks up any filter that is aligned with the two dominant directions corresponding to the repeated lines. Hence it can be seen that the banded image with a repeated linear structure is classified as a false positive grid image instead.

In contrast, the stained image contains irregular patterns with regions of varying regions and colour rather tahn a stable repeated orientation or frequency. This shows that while colour channels and low-frequency textures can provide some evidence, it is insufficient for Gabor orientations and edge-density features to capture due to the lack of consistent patterns. This results in false positive grid predictions as shown in the probability heat map.

Q6. The handcrafted global descriptor retains summary statistics of local colour, Gabor response, gradient magnitude, and edge-density responses. This indicates that it contains information about dominant orientations, repeated spatial frequencies, overall colour distributions, texture roughness, and average presence of edges. This makes it useful for attributes with consistent repeated structure, such as banded, striped, and grid.[AI-LANGUAGE][HUMAN-CHECK]

However, global pooling reduces each feature map to image-wide statistics, so it discarding spatial location of features. For example, it cannot reliably distinguish whether lines form a grid, occur only in one region, or follow a particular layout if their global feature distributions are similar. It also has limited ability to represent semantic appearance, object-level shape or changes in illumination. Consequently, textures with similar orientations or frequency content can be confused, as in the banded and grid examples.[AI-LANGUAGE][HUMAN-CHECK]

## Results and Analysis Task D
| Level | Information Supplied | Question being tested |
| --- | --- | --- |
| **D1: Generic zero-shot** | Test image and a generic instruction such as “Find any defect in this surface.” | Can the model discover that something is abnormal without domain guidance? |
| **D2: Named-defect zero-shot** | Test image, material name, and allowed defect names with short verbal definitions; no defect images | Does domain vocabulary help the model recognize and name defects |
| **D3: Example-conditioned** | D2 information plus the fixed labelled support set from public validation | Can visual examples help transfer a defect concept to new images? |

### Gemini 3.6 Flash
#### <u>Information Supplied</u>
**D1: Generic zero-shot**

<u>Information supplied</u>: Test image and prompt “Inspect this surface image. Is anything visibly abnormal? Return JSON with `is_defective`, `defect_name`, `confidence`, and `evidence`.”

**D2: Named-defect zero-shot**

<u>Information supplied</u>: Test image and prompt “Based on the provided material types and list of defect_names below, inspect this surface image. Is anything visibly abnormal? Return JSON with `is_defective`, `defect_name`, `confidence`, and `evidence`. If it is good, return good for defect name.

Q1 — carpet color: A localized region has an abnormal colour or tone. cut: A sharp incision or sliced region interrupts the texture. hole: Material is missing in a compact hole-like region. metal_contamination: A metallic foreign object lies on the surface. thread: A foreign thread lies across the regular texture.

Q2 — grid bent: The regular grid lines are locally warped or displaced. broken: Part of the repeating grid is interrupted or missing. glue: A localized glue deposit changes the surface appearance. metal_contamination: A metallic foreign object lies on the surface. thread: A foreign thread lies across the regular texture.

Q3 — leather color: A localized region has an abnormal colour or tone. cut: A sharp incision or sliced region interrupts the texture. fold: The surface is creased or folded out of its normal plane. glue: A localized glue deposit changes the surface appearance. poke: A small puncture-like indentation or opening is visible.

Q4 — tile crack: A narrow fracture-like line crosses the material surface. glue_strip: An elongated strip of adhesive lies on the tile. gray_stroke: A grey linear mark appears across the tile. oil: An oily region changes colour or reflectance. rough: A localized patch is visibly rougher than normal.

Q5 — wood color: A localized region has an abnormal colour or tone. combined: More than one defect mechanism appears in the same image. hole: Material is missing in a compact hole-like region. liquid: A liquid stain changes colour or reflectance locally. scratch: A thin elongated abrasion crosses the surface. 
”

**D3: Example-conditioned**

<u>Information supplied</u>: Test image, fixed support images, and exact same prompt as **D2**

#### <u>Model Responses</u>

The 15 responses below are frozen before consulting the answer key. The **Exact prompt** column points to the verbatim prompt stated in the *Information Supplied* section above: D1 is the generic prompt in D1; D2 is the full named-defect prompt (including Q1–Q5 definitions) in D2; and D3 uses that same D2 prompt plus its fixed labelled support set.[AI-LANGUAGE][HUMAN-CHECK]

| Query ID | Condition | Predicted label | Decision | Confidence | Visible evidence | Exact prompt | Model/version | Run date |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| Q1 (carpet) | D1: Generic zero-shot | stain | Defective | 0.98 | A dark, localized discoloration/stain is visible in the lower-left quadrant of the woven fabric texture. | D1 verbatim prompt above + Defect Image | Gemini 3.6 Flash | 06/10/2026 |
| Q1 (carpet) | D2: Named-defect zero-shot | color | Defective | 0.95 | There is a distinct dark localized stain/discoloration in the lower-left central quadrant of the carpet weave surface. | D2 verbatim prompt above + Defect Image | Gemini 3.6 Flash | 06/10/2026 |
| Q1 (carpet) | D3: Example-conditioned | color | Defective | 0.95 | A localized dark discoloration spot is visible near the center-left region of the carpet texture. | D2 verbatim prompt above + Defect Image + fixed labelled support set | Gemini 3.6 Flash | 06/10/2026 |
| Q2 (grid) | D1: Generic zero-shot | broken_strand | Defective | 0.98 | There are distinct structural breaks and bent/detached strands in the metal mesh, visibly disrupting the uniform diamond grid pattern in the upper-middle and lower regions. | D1 verbatim prompt above + Defect Image | Gemini 3.6 Flash | 06/10/2026 |
| Q2 (grid) | D2: Named-defect zero-shot | broken | Defective | 0.98 | Parts of the repeating metal grid structure are interrupted, severed, and missing in multiple areas (notably in the upper-middle and lower regions). | D2 verbatim prompt above + Defect Image | Gemini 3.6 Flash | 06/10/2026 |
| Q2 (grid) | D3: Example-conditioned | bent | Defective | 0.98 | The wire mesh grid lines are visibly deformed, bent, and distorted away from their regular repeating pattern in multiple localized areas across the image. | D2 verbatim prompt above + Defect Image + fixed labelled support set | Gemini 3.6 Flash | 06/10/2026 |
| Q3 (leather) | D1: Generic zero-shot | cut/tear | Defective | 0.98 | There is a visible cut or tear with exposed underlying fibers in the lower-left quadrant of the textured brown leather surface. | D1 verbatim prompt above + Defect Image | Gemini 3.6 Flash | 06/10/2026 |
| Q3 (leather) | D2: Named-defect zero-shot | poke | Defective | 0.98 | There is a small puncture-like opening/indentation in the lower-left region of the brown leather surface with exposed underlying material. | D2 verbatim prompt above + Defect Image | Gemini 3.6 Flash | 06/10/2026 |
| Q3 (leather) | D3: Example-conditioned | cut | Defective | 0.98 | A sharp incision line with exposed inner material fibers/edges is clearly visible in the lower-left quadrant of the leather surface. | D2 verbatim prompt above + Defect Image + fixed labelled support set | Gemini 3.6 Flash | 06/10/2026 |
| Q4 (tile) | D1: Generic zero-shot | crack | Defective | 0.99 | A distinct Y-shaped dark crack spans across the patterned surface, extending from the top left and top right down toward the bottom center. | D1 verbatim prompt above + Defect Image | Gemini 3.6 Flash | 06/10/2026 |
| Q4 (tile) | D2: Named-defect zero-shot | crack | Defective | 0.98 | A narrow fracture-like line splits into a Y-shape across the ceramic tile surface. | D2 verbatim prompt above + Defect Image | Gemini 3.6 Flash | 06/10/2026 |
| Q4 (tile) | D3: Example-conditioned | crack | Defective | 0.98 | Prominent dark Y-shaped crack lines fracturing the tile surface across the center. | D2 verbatim prompt above + Defect Image + fixed labelled support set | Gemini 3.6 Flash | 06/10/2026 |
| Q5 (wood) | D1: Generic zero-shot | crack | Defective | 0.98 | A prominent vertical crack running vertically down the right-center section of the wood surface. | D1 verbatim prompt above + Defect Image | Gemini 3.6 Flash | 06/10/2026 |
| Q5 (wood) | D2: Named-defect zero-shot | scratch | Defective | 0.95 | A prominent vertical linear line/abrasion running down the right-center portion of the wood surface. | D2 verbatim prompt above + Defect Image | Gemini 3.6 Flash | 06/10/2026 |
| Q5 (wood) | D3: Example-conditioned | scratch | Defective | 0.95 | A prominent vertical linear scratch runs from top to bottom slightly to the right of the center line, where the wood surface is clearly abraded and scored. | D2 verbatim prompt above + Defect Image + fixed labelled support set | Gemini 3.6 Flash | 06/10/2026 |

**Accuracy:** 0.733 \
Accuracy was calculated using the standard formula $\frac{TP + TN}{TP + TN + FP + FN}$. In this case, TN and FN are 0 as all input images are defective and all model classifications are positive. TP would be when the model identifies the right defect while FP would be when the model identifies the wrong defect. TP = 11, FP = 4

Vocabulary compliance was evaluated across D1, D2 and D3 responses. While it may be questionable as D1 was opened ended, we wanted to see how close the model would be able to label defects as the stated example. This was also an exmaple of prompt sensitivity as it shows the model often used free-form for D1 but semantically related terms such as `stain`, `broken_strand`, and `cut/tear`, whereas D2 and D3 were guided toward the supplied label vocabulary  If we evaluate on the strict condition that the evidence should be the exact sentence provided in the examples, then D2 produced 0/5 out-of-vocabulary labels, and D3 likewise produced 0/5. As such the evaluation was done based on matching labels with those that were stated in the example. The model remained within the stated vocabulary for Carpet D2/D3, Grid D2/D3, Leather D2/D3, Tile D1/D2/D3 and Wood D2/D3. So the number of answers that violate the provided vocabulary is 4, all of which are D1. Because MVTec is public, prior exposure during model training cannot be ruled out. However, the present qualitative probe provides no basis for determining whether such exposure occurred.

Nondeterminism was not measured because each image-condition combination was run once. Comparing D1, D2, and D3 does not test nondeterminism, since the prompts and supplied context differ. Measuring it would require repeated runs using the same image, exact prompt, model version, support set, and generation settings.

Examples appear to help for the grid query as D2 selected broken, while D3 changed the prediction to bent. Examples also appear to help for leather as D2 selected poke, while D3 selected cut, matching the expected label. For carpet, tile, and wood, D2 and D3 agree, so this small set does not isolate a clear example-conditioning effect for those queries.

## Results and Analysis Custom Photos

There were a total of 4 MVTec material classes (carpet, grid, tile, wood) spread across 17 different texture photos.Out of the 17 photos, only 6 were classified correctly. 

Correctly classified:
![Tile 3](student_files/personal_texture_passport/personal_tile3_tp.png)
![Wood 1](student_files/personal_texture_passport/personal_wood1_tp.png)
![Wood 3](student_files/personal_texture_passport/personal_wood3_tp.png)
![Wood 5](student_files/personal_texture_passport/personal_wood5_tp.png)
![Wood 6](student_files/personal_texture_passport/personal_wood6_tp.png)
![Wood 7](student_files/personal_texture_passport/personal_wood7_tp.png)

Wrongly classified:
![Carpet 1](student_files/personal_texture_passport/personal_carpet1_tp.png)
![Carpet 2](student_files/personal_texture_passport/personal_carpet2_tp.png)
![Grid 1](student_files/personal_texture_passport/personal_grid1_tp.png)
![Grid 2](student_files/personal_texture_passport/personal_grid2_tp.png)
![Grid 3](student_files/personal_texture_passport/personal_grid3_tp.png)
![Tile 1](student_files/personal_texture_passport/personal_tile1_tp.png)
![Tile 2](student_files/personal_texture_passport/personal_tile2_tp.png)
![Wood 2](student_files/personal_texture_passport/personal_wood2_tp.png)
![Wood 4](student_files/personal_texture_passport/personal_wood4_tp.png)
![Wood 8](student_files/personal_texture_passport/personal_wood8_tp.png)
![Wood 9](student_files/personal_texture_passport/personal_wood9_tp.png)

Among the 17 photos, only 6 matched its predicted material. One likely reason could be due to the fact that models were not trained on the textures of the personal set, hence they were unable to correctly classify the textures according to their MVTec materials. This is a qualitative transfer observation rather than an accuracy estimate as the personal carpet, cutting-mat grid, floor tiles and household wood photos differ substantially from the MVTec acquisition setting.

Illumination also affects the models ability to classify the correct material type, specifically low illumination. As seen from the texture passport of an image with low illumination. As compared to an image with similar texture but higher illumination (Wood 1), the image with low illumination was wrongly classified as leather instead of wood.

![Material Confidence vs Anomaly Score](student_files/personal_texture_passport/analysis/confidence_vs_anomaly_by_illumination.png)[AI-DESIGN] [HUMAN-CHECK]

Material confidence and anomaly score were weakly related. High confidence did not imply that the frozen normality model found an image typical. For example, grid2 and carpet2 both received material confidence of 1.00 while producing anomaly scores of 27.40 and 19.78, respectively. The plot therefore shows that the closed-set material confidence and anomaly score capture different model behaviours.

![Illumination, Confidence, Correctness](student_files/personal_texture_passport/analysis/illumination_confidence_correctness.png)[AI-DESIGN] [HUMAN-CHECK]

After plotting the mean confidence and correctness rate against illumination levels, it can be seen that higher illumination leads to higher model confidence and higher correctness rate. The main reason for this is likely that higher illumination leads to better definition and contrast of material and texture features, hence leading to a higher confidence and correctness rate.

![Viewpoint, Confidence, Correctness](student_files/personal_texture_passport/analysis/viewpoint_confidence_correctness.png)[AI-CODE] [HUMAN-DESIGN]

Based on this, the front-on viewpoint yielded the highest correctness rate (0.71) as well as mean confidence (0.97) as opposed to the other two viewpoints with oblique performing the worst in terms of correctness (0.0) although it only contained two images. This may indicate that the front-on viewpoint was more compatible and hence better able to display features for the specific material and attributes. For other materials howvever, this might not be the case.

![Illumination, Attribute Overlap](student_files/personal_texture_passport/analysis/illumination_attribute_overlap.png)[AI-CODE] [HUMAN-DESIGN]
![Viewpoint, Attribute Overlap](student_files/personal_texture_passport/analysis/viewpoint_attribute_overlap.png)[AI-CODE] [HUMAN-DESIGN]

Based on the two plots of illumination levels and Model-Human and Human-Human attribute overlap as well as viewpoints and Model-Human and Human-Human attribute overlap, Human–model Jaccard overlap varied only from 0.21 to 0.25 across illumination groups and from 0.21 to 0.27 across viewpoint groups. However, it cannot be inferred that these two factors do not have much of an effect on the attributes predicted by the attribute model. This could be due to other uncontrolled factors such as difference in resolution and materials tested.

All results stated in this analysis may not be completely representative of model behaviour under the stated circumstances due to the limitation of a small dataset.

## Implementation Evidence

| Student file | Required functions | Report section | Evidence location |
|---|---|---|---|
| `gabor_branch.py` | `make_gabor_bank`, `gabor_energy_maps` | Gabor Implementation | gabor_branch.py lines 10-110, Notebook 1 part3. Build and inspect the Gabor bank, part 4. Local response-energy maps, part 6. Assemble and audit the shared local descriptor |
| `edge_branch.py` | `bilinear_sample`, `nms_interpolated`, `adaptive_thresholds`, `hysteresis`, `detect_edges` | Edge Implementation | Notebook 1, [replace with exact section/figure]: magnitude, NMS, and final-edge-map evidence; report the selected low/high thresholds. |
| `features.py` | `extract_local_features`, `global_pool`, `feature_family_indices` | Features and Representations | features.py lines 11-200, Notebook 1 part 6. Assemble and audit the shared local descriptor, Notebook 2, part 3. Task A — material classification |
| `normality.py` | `fit_normal_model`, `predict_anomaly`, `select_mask_threshold` | Normality Model | Notebook 2: anomaly score-map and mask, validation-selected threshold, and detection/segmentation metrics. |

## AI-use disclosure table

| Marker | Meaning |
|---|---|
| `[AI-CODE]` | AI generated, debugged, translated, or refactored code. |
| `[AI-DESIGN]` | AI created plots and images according to student provided direction. |
| `[AI-LANGUAGE]` | An LLM transformed points provided by a student into coherent report-style sentences/paragraphs. |
| `[HUMAN-CHECK]` | A student verified, rejected, corrected, or stress-tested an AI contribution. |
| `[HUMAN-DESGIN]` | A student proposed hypothesis, experiments or metrics. |

AI Model Used ChatGPT (GPT-5.6 Sol and Codex (GPT-5.6-Terra max)
AI was used throughout this project to understand the project, carry out laborious tasks, such as markdown table generation and formatting.
For every piece of code written by AI, the student read through and verified it based on the guiding comments provided.
