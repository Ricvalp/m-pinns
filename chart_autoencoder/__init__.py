from chart_autoencoder.trainer import Trainer
from chart_autoencoder.trainer_autodecoder import TrainerAutoDecoder
from chart_autoencoder.trainer_autoencoder import TrainerAutoEncoder, ChartsDataset
from chart_autoencoder.plot import (
    plot_3d_points,
    plot_3d_charts,
    plot_3d_boundaries,
    plot_local_charts_2d,
    plot_local_charts_2d_with_boundaries,
    plot_html_3d_point_cloud,
    plot_html_3d_charts,
    plot_html_3d_boundaries,
)
from chart_autoencoder.utils import (
    ChartSampler,
    ModelCheckpoint,
    numpy_collate,
    numpy_collate_with_distances,
    prefetch_to_device,
    get_model,
    compute_distance_matrix,
)
from chart_autoencoder.models import AutoEncoder
from chart_autoencoder.get_charts import (
    get_charts,
    load_charts,
    load_charts3d,
    save_charts,
    refine_chart,
    reindex_charts,
    find_intersection_indices,
    find_verts_in_charts,
    find_closest_points_to_mesh,
    fast_region_growing,
)
from chart_autoencoder.umap_embedding import get_umap_embeddings
from chart_autoencoder.riemann import (
    compute_norm_g_ginv_from_params,
    get_metric_tensor_and_sqrt_det_g,
    get_metric_tensor_and_sqrt_det_g_autodecoder,
    compute_norm_g_ginv_from_params_autoencoder,
    get_metric_tensor_and_sqrt_det_g_universal_autodecoder,
)
from chart_autoencoder.persistence_homology_utils import compute_persistence_homology
