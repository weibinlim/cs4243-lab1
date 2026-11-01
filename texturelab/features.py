"""Aligned local descriptor shared by all three tasks."""

from __future__ import annotations
import numpy as np
from .config import FeatureConfig
from .edge_branch import detect_edges
from .gabor_branch import gabor_energy_maps, make_gabor_bank
from .supplied import box_mean, rgb2gray


def extract_local_features(image: np.ndarray, config: FeatureConfig) -> tuple[np.ndarray, list[str]]:
    """Return an aligned H x W x D map and one name per channel."""
    # YOUR CODE HERE
    #
    # TODO 1 — Normalise the input representation
    #   - Convert to float32; expand a 2-D image to three identical channels.
    #   - Scale 0--255-style inputs to 0--1 while leaving 0--1 inputs unchanged.
    #   - Compute grayscale once with rgb2gray.

    img = image.astype(np.float32, copy=False)
    if img.ndim == 2: img = np.stack([img] * 3, axis=-1)
    if img.max() > 1: img = img / 255.0
    gray = rgb2gray(img).astype(np.float32, copy=False)

    # TODO 2 — Initialise aligned channel/name groups
    #   - Keep feature chunks and their names in matching, deterministic order.
    #   - If include_colour, append the first three image channels named
    #     colour_r, colour_g, and colour_b.
    #
    channels = []
    names = []
    if config.include_colour:
        channels.append(img[:, :, :3])
        names.extend(["colour_r", "colour_g", "colour_b"])

    # TODO 3 — Add the Gabor branch when enabled
    #   - Build config.gabor's bank and compute pooled maps using its pool_size
    #     and energy mode.
    #   - Create one stable name from each kernel's frequency/orientation/phase.
    
    if config.include_gabor:
        bank = make_gabor_bank(config.gabor)
        gabor_maps = gabor_energy_maps(
            gray,
            bank,
            pool_size=config.gabor.pool_size,
            energy=config.gabor.energy,
        )
        channels.append(gabor_maps)

        for kernel, meta in bank:
            freq = meta["frequency"]
            ori = meta["orientation"]
            ph = meta["phase"]
            names.append(f"gabor_f{freq}_o{ori}_p{ph}")

    # TODO 4 — Compute edge data only when needed
    #   - Call detect_edges once if gradient or edge channels are requested.
    #   - For gradient energy, square magnitude, box-pool it with density_size,
    #     add a singleton channel axis, and name it gradient_energy.
    #
    if config.include_edges or config.include_gradient:
        edge_data = detect_edges(gray, config.edge)

        if config.include_gradient:
            gradient_energy = np.square(edge_data["magnitude"])
            gradient_energy = box_mean(
                gradient_energy,
                config.edge.density_size
            )

            channels.append(gradient_energy[..., None])
            names.append("gradient_energy")

    # TODO 5 — Add total and orientation-specific edge density
    #   - Pool the Boolean edge map for edge_density.
    #   - Map directions modulo pi, divide the unsigned orientation range into
    #     config.edge.n_orientations bins, and pool edges belonging to each bin.
    #   - Name bins edge_orientation_0, edge_orientation_1, and so on.
    #
        if config.include_edges:
            edge_density = box_mean(
                edge_data["edges"].astype(np.float32),
                config.edge.density_size
            )

            channels.append(edge_density[..., None])
            names.append("edge_density")

            orientation = np.mod(edge_data["direction"], np.pi)

            n_bins = config.edge.n_orientations
            bin_width = np.pi / n_bins

            for i in range(n_bins):
                lower = i * bin_width
                upper = (i + 1) * bin_width

                orientation_edges = (
                    edge_data["edges"]
                    & (orientation >= lower)
                    & (orientation < upper)
                ).astype(np.float32)

                orientation_density = box_mean(
                    orientation_edges,
                    config.edge.density_size
                )

                channels.append(orientation_density[..., None])
                names.append(f"edge_orientation_{i}")


    # TODO 6 — Assemble and optionally standardise
    #   - Reject a configuration with no enabled feature family.
    #   - Concatenate chunks along the final axis and return float32.
    #   - If standardise_per_image, standardise every channel over H/W using a
    #     small epsilon; return the feature map and equally long name list.

    if not channels:
        raise ValueError("At least one feature family must be enabled.")

    features = np.concatenate(channels, axis=-1).astype(np.float32, copy=False)

    if config.standardise_per_image:
        eps = 1e-6

        mean = np.mean(features, axis=(0, 1), keepdims=True)
        std = np.std(features, axis=(0, 1), keepdims=True)

        features = (features - mean) / (std + eps)

        features = features.astype(np.float32, copy=False)

    if features.shape[-1] != len(names):
        raise ValueError("Feature channel count does not match number of names.")

    return features, names


def global_pool(feature_map: np.ndarray,
                statistics: tuple[str, ...] = ("mean", "std", "p90")) -> np.ndarray:
    """Pool local channels into one reproducible image descriptor."""
    # YOUR CODE HERE
    #
    # TODO 1 — Flatten spatial positions
    #   - Reshape H x W x D into (H*W) x D without changing channel order.
    #
    height, width, depth = feature_map.shape
    flat = feature_map.reshape(height * width, depth)

    # TODO 2 — Implement the supported statistics
    #   - Support mean, std, p10, p50, and p90, each computed per channel.
    #   - Raise ValueError naming an unsupported statistic.
    #
    pooled = []

    for stat in statistics:
        if stat == "mean": values = np.mean(flat, axis=0)
        elif stat == "std": values = np.std(flat, axis=0)
        elif stat == "p10": values = np.percentile(flat, 10, axis=0)
        elif stat == "p50": values = np.percentile(flat, 50, axis=0)
        elif stat == "p90": values = np.percentile(flat, 90, axis=0)
        else: raise ValueError(f"Unsupported statistic: {stat}")

        pooled.append(values)

    # TODO 3 — Assemble the descriptor
    #   - Evaluate statistics in the exact caller-supplied order.
    #   - Concatenate their D-vectors and return a one-dimensional float32 array.

    descriptor = np.concatenate(pooled).astype(np.float32, copy=False)
    return descriptor

def feature_family_indices(names: list[str]) -> dict[str, list[int]]:
    """Map colour/Gabor/gradient/edge families to descriptor indices."""
    # YOUR CODE HERE
    #
    # TODO 1 — Create all required keys
    #   - Initialise colour, gabor, gradient, and edge to empty index lists.
    #
    families = {
        "colour": [],
        "gabor": [],
        "gradient": [],
        "edge": [],
    }

    # TODO 2 — Assign channels by their stable name prefix
    #   - Iterate through names in order, take the text before the first "_",
    #     and append the channel index to the matching family.
    #   - Return every family key even when that family has no channels.
    for i, name in enumerate(names):
        prefix = name.split("_", 1)[0]

        if prefix in families:
            families[prefix].append(i)

    return families
