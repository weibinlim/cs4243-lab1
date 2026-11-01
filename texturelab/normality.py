"""Per-material diagonal normal model and dense anomaly prediction."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import NormalityConfig
from .supplied import box_mean


@dataclass
class NormalModel:
    mean: np.ndarray
    std: np.ndarray
    threshold: float
    feature_names: list[str] | None = None


def _scores(
    features: np.ndarray, mean: np.ndarray, std: np.ndarray, epsilon: float
) -> np.ndarray:
    return np.sqrt(np.mean(((features - mean) / (std + epsilon)) ** 2, axis=-1)).astype(
        np.float32
    )


def _pool_score(score: np.ndarray, config: NormalityConfig) -> np.ndarray:
    """Suppress isolated patch responses while preserving coherent anomalies."""
    if config.score_pool_size < 1 or config.score_pool_size % 2 == 0:
        raise ValueError("score_pool_size must be a positive odd integer")
    return box_mean(score, config.score_pool_size).astype(np.float32)


def _interior(score: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return score
    if 2 * border >= min(score.shape):
        raise ValueError("ignore_border is too large for the score map")
    return score[border:-border, border:-border]


def fit_normal_model(feature_maps: list[np.ndarray] | np.ndarray, config: NormalityConfig,
                     feature_names: list[str] | None = None) -> NormalModel:
    """Fit normal feature statistics using normal training maps only."""
    # YOUR CODE HERE
    #
    # TODO 1 — Normalise and validate the collection
    #   - Accept either one H x W x D array or a non-empty collection of maps.
    #   - Reject an empty collection; preserve the common feature dimension.
    
    if isinstance(feature_maps, np.ndarray):
        maps = [feature_maps]
    else:
        if feature_maps is None or len(feature_maps) == 0:
            raise ValueError("Feature maps is empty")
        maps = feature_maps

    for i, feature_map in enumerate(maps):
        if not isinstance(feature_map, np.ndarray):
            raise TypeError(f"Feature map {i} must be a NumPy array.")
        if feature_map.ndim != 3:
            raise ValueError(f"Feature map {i} must have shape (H, W, D); got {feature_map.shape}.")

    feature_dim = maps[0].shape[2]

    if any(feature_map.shape[2] != feature_dim for feature_map in maps):
        raise ValueError("All feature maps must have the same feature dimension D.")
    
    # TODO 2 — Build the normal patch sample
    #   - Flatten each map to (H*W) x D and concatenate across training images.
    #   - If there are more than config.max_samples rows, select exactly that
    #     many without replacement using config.random_seed.

    patches_per_image = [feature_map.reshape(-1, feature_dim) for feature_map in maps]
    all_patches = np.concatenate(patches_per_image, axis=0)
    if all_patches.shape[0] > config.max_samples:
        rng = np.random.default_rng(config.random_seed)
        selected_indices = rng.choice(all_patches.shape[0], size=config.max_samples, replace=False)
        all_patches = all_patches[selected_indices]
    
    # TODO 3 — Fit diagonal normal statistics
    #   - Compute a D-vector mean and population standard deviation.
    
    d_mean = all_patches.mean(axis=0)
    d_std = all_patches.std(axis=0)

    # TODO 4 — Estimate a normal-only provisional threshold
    #   - Score every original map with _scores, smooth it with _pool_score, crop
    #     the configured unreliable border with _interior, then concatenate.
    #   - Use config.mask_threshold when explicitly supplied; otherwise take
    #     config.threshold_percentile of the pooled interior normal scores.
    
    all_cropped_scores = []
    all_scores = None
    for fmap in maps:
        score = _scores(fmap, d_mean, d_std, config.epsilon)
        smooth_score = _pool_score(score, config)
        croppped_score = _interior(smooth_score, config.ignore_border)
        all_cropped_scores.append(croppped_score)
        all_scores = np.concatenate(all_cropped_scores)

    if config.mask_threshold is None:
        threshold = np.percentile(all_scores, config.threshold_percentile)
    else:
        threshold = config.mask_threshold

    # TODO 5 — Return the model
    #   - Store float32 mean/std, a Python-float threshold, and feature_names.

    return NormalModel(
        mean=d_mean.astype(np.float32), 
        std=d_std.astype(np.float32), 
        threshold=threshold.astype(np.float32),
        feature_names=feature_names
    )


def predict_anomaly(feature_map: np.ndarray, model: NormalModel, config: NormalityConfig
                    ) -> tuple[np.ndarray, float, np.ndarray]:
    """Return dense score, image-level score, and predicted mask."""
    # YOUR CODE HERE
    #
    # TODO 1 — Validate feature compatibility
    #   - The final feature dimension must equal the fitted model mean length.

    if feature_map.ndim != 3:
        raise ValueError("Feature_map must have shape (H, W, D)")

    if feature_map.shape[-1] != len(model.mean):
        raise ValueError("Feature dimension does not match the fitted normal model.")
    
    # TODO 2 — Compute the dense anomaly score
    #   - Use _scores for RMS standardized distance with config.epsilon.
    #   - Smooth coherent evidence with _pool_score.
    
    rms_stan_dist = _scores(feature_map, model.mean, model.std, config.epsilon)
    smooth_rms_stan_dist = _pool_score(rms_stan_dist, config)

    # TODO 3 — Aggregate an image-level score
    #   - Crop the unreliable border with _interior and take
    #     config.image_percentile over that interior only.
    
    cropped = _interior(smooth_rms_stan_dist, config.ignore_border)
    image_score = float(np.percentile(cropped, config.image_percentile))

    # TODO 4 — Produce the binary localization mask
    #   - Threshold the full score map at model.threshold.
    #   - Force the configured outer border to False without altering scores.
    #   - Return (float32 score map, Python-float image score, Boolean mask).

    mask = smooth_rms_stan_dist >= model.threshold
    border = config.ignore_border
    if border > 0:
        if 2 * border >= min(smooth_rms_stan_dist.shape):
            raise ValueError("ignore_border is too large for the score map")

        mask[:border, :] = False
        mask[-border:, :] = False
        mask[:, :border] = False
        mask[:, -border:] = False

    return smooth_rms_stan_dist.astype(np.float32), image_score, mask.astype(bool)


def select_mask_threshold(score_maps: list[np.ndarray], masks: list[np.ndarray],
                          candidates: int = 80) -> float:
    """Choose the pixel-F1-optimal threshold on public validation only."""
    # YOUR CODE HERE
    #
    # TODO 1 — Validate the validation inputs
    #   - Require matching non-empty score/mask collections and candidates >= 2.
    #   - Convert scores to float and masks to bool; require matching shapes and
    #     finite scores for every pair.
    #   - Callers handle border cropping, so do not crop again here.
    
    if score_maps is None or len(score_maps) == 0 or masks is None or len(masks) == 0:
        raise ValueError("Score Maps or Masks cannot be empty.")

    if len(score_maps) != len(masks):
        raise ValueError("Score Map and Masks are not matching.")

    if candidates < 2:
        raise ValueError("Candidates cannot be less than 2 for select_mask_threshold.")

    score_maps_float = [np.asarray(score, dtype=float) for score in score_maps]
    masks_bool = [np.asarray(mask, dtype=bool) for mask in masks]

    for score, mask in zip(score_maps_float, masks_bool):
        if score.shape != mask.shape:
            print(
                "score:", score.shape,
                "mask:", mask.shape,
            )
            raise ValueError("Each score mpa must have the same shape as its mask.")

        if not np.isfinite(score).all():
            raise ValueError("Score maps must contain only finite values.")

    # TODO 2 — Build candidate thresholds
    #   - Flatten and concatenate all validation pixels.
    #   - Use evenly spaced quantile levels from 0.5 through 0.999.
    
    all_scores = np.concatenate([score.ravel() for score in score_maps_float])
    all_masks = np.concatenate([mask.ravel() for mask in masks_bool])
    levels = np.linspace(0.5, 0.999, candidates)
    thresholds = np.quantile(all_scores, levels)

    # TODO 3 — Select by pixel F1
    #   - For each threshold, predict score >= threshold and compute TP/FP/FN.
    #   - Use 2*TP / max(1, 2*TP + FP + FN) to avoid division by zero.
    #   - Keep the threshold with the greatest F1; deterministic ties should
    #     retain the first encountered candidate.
    #   - Return the selected threshold as a Python float.

    best_t = (0, float(np.inf))
    for threshold in thresholds:
        prediction = all_scores >= threshold
        tp = np.sum(prediction & all_masks)
        fp = np.sum(prediction & ~all_masks)
        fn = np.sum(~prediction & all_masks)

        f1 = (2 * tp) / max(1, 2 * tp + fp + fn)

        if f1 > best_t[0]:
            best_t = (f1, threshold)

    return float(best_t[1])
