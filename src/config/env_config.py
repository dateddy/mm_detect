"""Environment configuration management"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv


class EnvConfig:
    """Load and manage environment variables from .env file."""

    def __init__(self, env_file: Optional[Path] = None):
        """
        Initialize environment configuration.

        Args:
            env_file: Path to .env file. If None, searches in project root.
        """
        if env_file is None:
            # Look for .env in project root
            project_root = Path(__file__).parent.parent.parent
            env_file = project_root / ".env"

        self.env_file = Path(env_file)

        # Load environment variables from .env file when present. Most training
        # and ablation scripts use YAML configs only, so missing .env should not
        # make importing src.config.schema fail.
        if self.env_file.exists():
            load_dotenv(self.env_file)

    # ========== Data Paths ==========

    @property
    def data_dir(self) -> Path:
        """Root data directory."""
        return Path(os.getenv("DATA_DIR", "data"))

    @property
    def data_raw_dir(self) -> Path:
        """Raw data directory."""
        return Path(os.getenv("RAW_DATA_DIR", "data/raw"))

    @property
    def data_processed_dir(self) -> Path:
        """Processed data directory."""
        return Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))

    @property
    def embeddings_dir(self) -> Path:
        """Embeddings directory."""
        return Path(os.getenv("EMBEDDINGS_DIR", "data/embeddings"))

    @property
    def images_dir(self) -> Path:
        """Images directory."""
        return Path(os.getenv("IMAGES_DIR", "data/raw/ad_images"))

    # ========== Config Paths ==========

    @property
    def model_config_dir(self) -> Path:
        """Model config directory."""
        return Path(os.getenv("MODEL_CONFIG_DIR", "configs/model"))

    @property
    def training_config_dir(self) -> Path:
        """Training config directory."""
        return Path(os.getenv("TRAINING_CONFIG_DIR", "configs/training"))

    @property
    def base_config(self) -> Path:
        """Base config file."""
        return Path(os.getenv("BASE_CONFIG", "configs/base.yaml"))

    @property
    def default_model_config(self) -> Path:
        """Default model config file."""
        return Path(os.getenv("DEFAULT_MODEL_CONFIG", "configs/model/multimodal.yaml"))

    @property
    def default_training_config(self) -> Path:
        """Default training config file."""
        return Path(
            os.getenv("DEFAULT_TRAINING_CONFIG", "configs/training/default.yaml")
        )

    # ========== Output Paths ==========

    @property
    def checkpoint_dir(self) -> Path:
        """Checkpoint directory."""
        return Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))

    @property
    def output_dir(self) -> Path:
        """Output directory."""
        return Path(os.getenv("OUTPUT_DIR", "outputs"))

    @property
    def logs_dir(self) -> Path:
        """Logs directory."""
        return Path(os.getenv("LOGS_DIR", "logs"))

    @property
    def figures_dir(self) -> Path:
        """Figures directory."""
        return Path(os.getenv("FIGURES_DIR", "outputs/figures"))

    # ========== Model Names ==========

    @property
    def text_model_name(self) -> str:
        """Text encoder model name."""
        return os.getenv("TEXT_MODEL_NAME", "vinai/phobert-base")

    @property
    def image_model_name(self) -> str:
        """Image encoder model name."""
        return os.getenv("IMAGE_MODEL_NAME", "google/vit-base-patch16-224")

    # ========== Training Parameters ==========

    @property
    def batch_size(self) -> int:
        """Batch size."""
        return int(os.getenv("BATCH_SIZE", "32"))

    @property
    def learning_rate(self) -> float:
        """Learning rate."""
        return float(os.getenv("LEARNING_RATE", "0.001"))

    @property
    def num_epochs(self) -> int:
        """Number of epochs."""
        return int(os.getenv("NUM_EPOCHS", "20"))

    @property
    def early_stopping_patience(self) -> int:
        """Early stopping patience."""
        return int(os.getenv("EARLY_STOPPING_PATIENCE", "5"))

    @property
    def gradient_clip(self) -> float:
        """Gradient clipping value."""
        return float(os.getenv("GRADIENT_CLIP", "1.0"))

    @property
    def dropout_rate(self) -> float:
        """Dropout rate."""
        return float(os.getenv("DROPOUT_RATE", "0.1"))

    # ========== Model Architecture ==========

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension."""
        return int(os.getenv("EMBEDDING_DIM", "768"))

    @property
    def num_heads(self) -> int:
        """Number of attention heads."""
        return int(os.getenv("NUM_HEADS", "8"))

    @property
    def hidden_dim(self) -> int:
        """Hidden dimension."""
        return int(os.getenv("HIDDEN_DIM", "256"))

    @property
    def num_classes(self) -> int:
        """Number of output classes."""
        return int(os.getenv("NUM_CLASSES", "1"))

    # ========== Optimization ==========

    @property
    def optimizer(self) -> str:
        """Optimizer type."""
        return os.getenv("OPTIMIZER", "Adam")

    @property
    def weight_decay(self) -> float:
        """Weight decay."""
        return float(os.getenv("WEIGHT_DECAY", "0.0001"))

    @property
    def scheduler(self) -> str:
        """Learning rate scheduler type."""
        return os.getenv("SCHEDULER", "cosine")

    @property
    def warmup_epochs(self) -> int:
        """Warmup epochs."""
        return int(os.getenv("WARMUP_EPOCHS", "2"))

    # ========== Loss Function ==========

    @property
    def loss_function(self) -> str:
        """Loss function type."""
        return os.getenv("LOSS_FUNCTION", "FocalLoss")

    @property
    def focal_alpha(self) -> float:
        """Focal loss alpha."""
        return float(os.getenv("FOCAL_ALPHA", "0.25"))

    @property
    def focal_gamma(self) -> float:
        """Focal loss gamma."""
        return float(os.getenv("FOCAL_GAMMA", "2.0"))

    @property
    def positive_weight(self) -> float:
        """Positive class weight."""
        return float(os.getenv("POSITIVE_WEIGHT", "1.0"))

    # ========== Data Processing ==========

    @property
    def text_max_length(self) -> int:
        """Maximum text length."""
        return int(os.getenv("TEXT_MAX_LENGTH", "256"))

    @property
    def image_size(self) -> int:
        """Image size."""
        return int(os.getenv("IMAGE_SIZE", "224"))

    @property
    def train_size(self) -> float:
        """Training set ratio."""
        return float(os.getenv("TRAIN_SIZE", "0.70"))

    @property
    def val_size(self) -> float:
        """Validation set ratio."""
        return float(os.getenv("VAL_SIZE", "0.15"))

    @property
    def test_size(self) -> float:
        """Test set ratio."""
        return float(os.getenv("TEST_SIZE", "0.15"))

    @property
    def random_seed(self) -> int:
        """Random seed."""
        return int(os.getenv("RANDOM_SEED", "42"))

    # ========== Device ==========

    @property
    def device(self) -> str:
        """Device type."""
        return os.getenv("DEVICE", "cuda")

    @property
    def use_cuda(self) -> bool:
        """Whether to use CUDA."""
        return os.getenv("USE_CUDA", "true").lower() == "true"

    @property
    def num_workers(self) -> int:
        """Number of workers for data loading."""
        return int(os.getenv("NUM_WORKERS", "4"))

    @property
    def pin_memory(self) -> bool:
        """Whether to pin memory."""
        return os.getenv("PIN_MEMORY", "true").lower() == "true"

    # ========== Logging ==========

    @property
    def log_level(self) -> str:
        """Console logging level."""
        return os.getenv("LOG_LEVEL", "INFO")

    @property
    def log_file_level(self) -> str:
        """File logging level."""
        return os.getenv("LOG_FILE_LEVEL", "DEBUG")

    # ========== Evaluation ==========

    @property
    def primary_metric(self) -> str:
        """Primary evaluation metric."""
        return os.getenv("PRIMARY_METRIC", "f1")

    @property
    def eval_threshold(self) -> float:
        """Evaluation threshold."""
        return float(os.getenv("EVAL_THRESHOLD", "0.5"))

    @property
    def compute_roc_auc(self) -> bool:
        """Whether to compute ROC-AUC."""
        return os.getenv("COMPUTE_ROC_AUC", "true").lower() == "true"

    # ========== Ablation ==========

    @property
    def ablation_epochs(self) -> int:
        """Number of epochs for ablation studies."""
        return int(os.getenv("ABLATION_EPOCHS", "20"))

    @property
    def ablation_patience(self) -> int:
        """Early stopping patience for ablation studies."""
        return int(os.getenv("ABLATION_PATIENCE", "3"))

    @property
    def ablation_variants(self) -> list:
        """List of ablation variants to test."""
        variants_str = os.getenv(
            "ABLATION_VARIANTS",
            "text_only,image_only,metadata_only,text_image,full_multimodal",
        )
        return [v.strip() for v in variants_str.split(",")]

    # ========== Debug ==========

    @property
    def debug(self) -> bool:
        """Whether to enable debug mode."""
        return os.getenv("DEBUG", "false").lower() == "true"

    @property
    def verbose(self) -> bool:
        """Whether to enable verbose logging."""
        return os.getenv("VERBOSE", "true").lower() == "true"

    @property
    def save_intermediate_results(self) -> bool:
        """Whether to save intermediate results."""
        return os.getenv("SAVE_INTERMEDIATE_RESULTS", "true").lower() == "true"

    # ========== Utility Methods ==========

    def to_dict(self) -> Dict[str, Any]:
        """Convert all config values to dictionary."""
        return {
            # Paths
            "data_dir": str(self.data_dir),
            "data_raw_dir": str(self.data_raw_dir),
            "data_processed_dir": str(self.data_processed_dir),
            "embeddings_dir": str(self.embeddings_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "output_dir": str(self.output_dir),
            "logs_dir": str(self.logs_dir),
            # Training
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            # Model
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
        }


# Global config instance
config = EnvConfig()
