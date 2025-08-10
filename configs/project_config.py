"""
Configuration for Self-Supervised Multimodal Learning Project

"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class ProjectConfig:

    # Project metadata
    project_name: str = "Self-Supervised Multimodal Mental Health Crisis Detection"
    version: str = "1.0.0"
    author: str = "Assignment 7 - High Risk Project"
    
    # Data paths
    mimic_data_path: str = "/Users/dutts/Desktop/Saj/_Postgraduate/MSAI/2. AI in Healthcare/Assignments/data/mimic-iv"
    project_data_path: str = "data"
    models_path: str = "models"
    results_path: str = "results"
    
    # Data processing parameters
    sample_size: Optional[int] = None  # None for full dataset, or number for testing
    min_note_length: int = 100
    max_text_length: int = 512
    
    # Mental health phenotyping
    crisis_prediction_windows: Dict[str, int] = None
    
    def __post_init__(self):
        if self.crisis_prediction_windows is None:
            self.crisis_prediction_windows = {
                'immediate': 1,      # 1 day
                'short_term': 7,     # 1 week
                'medium_term': 14,   # 2 weeks
                'long_term': 28      # 4 weeks (target)
            }

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Embedding dimensions
    text_embedding_dim: int = 768
    vitals_embedding_dim: int = 128
    meds_embedding_dim: int = 64
    fusion_hidden_dim: int = 512
    output_dim: int = 256
    
    # Self-supervised learning parameters
    temperature: float = 0.1  # For contrastive learning
    mask_probability: float = 0.15  # For masked autoencoding
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    validation_split: float = 0.2
    
    # Model architecture
    use_pretrained_text: bool = True
    text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"  # Clinical BERT
    lstm_hidden_dim: int = 256
    num_lstm_layers: int = 2
    dropout_rate: float = 0.1

@dataclass
class PrivacyConfig:
    """Privacy and security configuration."""
    
    # Differential privacy
    enable_differential_privacy: bool = True
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    privacy_budget: float = 10.0
    
    # Data minimization
    anonymize_text: bool = True
    remove_patient_identifiers: bool = True
    
    # Audit and compliance
    log_data_access: bool = True
    require_human_oversight: bool = True

@dataclass
class ExperimentConfig:
    """Experiment and evaluation configuration."""
    
    # Experiment tracking
    use_wandb: bool = False  # Set to True if using Weights & Biases
    experiment_name: str = "ssl_mental_health_v1"
    
    # Evaluation metrics
    primary_metric: str = "auroc"
    early_stopping_patience: int = 10
    
    # Cross-validation
    cv_folds: int = 5
    stratify_by: str = "crisis_target"
    
    # Ablation studies
    run_ablation_studies: bool = True
    ablation_components: List[str] = None
    
    def __post_init__(self):
        if self.ablation_components is None:
            self.ablation_components = [
                'text_only',
                'vitals_only', 
                'meds_only',
                'text_vitals',
                'text_meds',
                'vitals_meds',
                'all_modalities'
            ]

@dataclass
class EthicsConfig:
    """Ethical considerations and bias mitigation."""
    
    # Bias monitoring
    monitor_demographic_bias: bool = True
    fairness_metrics: List[str] = None
    protected_attributes: List[str] = None
    
    # Clinical deployment
    require_clinician_review: bool = True
    uncertainty_threshold: float = 0.1
    max_false_positive_rate: float = 0.1
    
    # Transparency
    provide_explanations: bool = True
    explanation_method: str = "shap"
    
    def __post_init__(self):
        if self.fairness_metrics is None:
            self.fairness_metrics = [
                'demographic_parity',
                'equalized_odds',
                'calibration'
            ]
        
        if self.protected_attributes is None:
            self.protected_attributes = [
                'gender',
                'age_group',
                'ethnicity'
            ]

# Create default configuration instances
PROJECT_CONFIG = ProjectConfig()
MODEL_CONFIG = ModelConfig()
PRIVACY_CONFIG = PrivacyConfig()
EXPERIMENT_CONFIG = ExperimentConfig()
ETHICS_CONFIG = EthicsConfig()

# High-risk project success criteria
HIGH_RISK_SUCCESS_CRITERIA = {
    'technical_innovation': [
        "Novel self-supervised learning approach for mental health",
        "Successful multimodal fusion of clinical data",
        "Privacy-preserving training mechanisms",
        "Early prediction capability (2-4 weeks)"
    ],
    'methodological_rigor': [
        "Comprehensive evaluation framework",
        "Proper temporal validation",
        "Bias analysis and fairness assessment",
        "Ablation studies and failure analysis"
    ],
    'clinical_relevance': [
        "Actionable early warning system",
        "Integration with clinical workflow",
        "Uncertainty quantification",
        "Human-in-the-loop design"
    ],
    'ethical_framework': [
        "Privacy protection mechanisms",
        "Bias mitigation strategies",
        "Transparent decision making",
        "Patient autonomy preservation"
    ],
    'research_impact': [
        "Open-source framework",
        "Reproducible methodology",
        "Clear documentation of limitations",
        "Foundation for future research"
    ]
}

# High-risk factors (why this project is challenging)
HIGH_RISK_FACTORS = {
    'technical_risks': [
        "Self-supervised learning for mental health is unproven",
        "Multimodal fusion is extremely complex",
        "Early prediction requires long temporal modeling",
        "Limited labeled crisis data",
        "Evaluation without ground truth is challenging"
    ],
    'domain_risks': [
        "Mental health patterns are highly individual",
        "Crisis definition is subjective and variable",
        "External factors (life events) confound patterns",
        "Clinical data is noisy and incomplete",
        "Temporal relationships are complex"
    ],
    'ethical_risks': [
        "Mental health data is highly sensitive",
        "False positives could harm patients",
        "Bias could discriminate against vulnerable groups",
        "Deployment could over-medicalize normal behavior",
        "Privacy violations could have severe consequences"
    ],
    'implementation_risks': [
        "MIMIC-IV data may not have sufficient mental health cases",
        "Required computational resources are substantial",
        "Clinical integration is complex",
        "Regulatory approval would be extensive",
        "Maintenance and updates are challenging"
    ]
}

def get_config_summary():
    """Get a summary of all configurations."""
    return {
        'project': PROJECT_CONFIG,
        'model': MODEL_CONFIG,
        'privacy': PRIVACY_CONFIG,
        'experiment': EXPERIMENT_CONFIG,
        'ethics': ETHICS_CONFIG,
        'success_criteria': HIGH_RISK_SUCCESS_CRITERIA,
        'risk_factors': HIGH_RISK_FACTORS
    }
