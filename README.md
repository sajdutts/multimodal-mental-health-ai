# Self-Supervised Multimodal Learning for Early Mental Health Crisis Detection

## Abstract

This project implements a self-supervised multimodal learning framework for early detection of mental health crises using MIMIC-IV clinical data. The approach combines clinical text, vital signs, and medication data through privacy-preserving self-supervised learning to predict mental health deterioration 2-4 weeks before clinical manifestation.

## Research Question

Can self-supervised multimodal learning detect patterns indicative of impending mental health crises 2-4 weeks before clinical manifestation, using privacy-preserving techniques?

**How do we define mental health criticality?**

**High-Risk (Critical)**: Imminent crisis indicators – suicidal ideation, psychosis, severe agitation

**Medium-Risk (Moderate)**: Deteriorating patterns – medication non-compliance, increasing symptoms

**Low-Risk (Stable)**: Baseline functioning – stable mood, medication adherence, normal activities


## Methodology

### Data Processing
- **Dataset**: MIMIC-IV clinical database with synthetic data
- **Population**: 9,151 patients with mental health conditions
- **Modalities**: Clinical notes, vital signs, medications
- **Preprocessing**: Advanced NLP, temporal windowing, medication analysis

### Model Architecture
- **Framework**: Self-supervised multimodal learning
- **Text Encoder**: Clinical BERT (Bio_ClinicalBERT)
- **Temporal Encoder**: Bidirectional LSTM
- **Fusion Layer**: Multi-head attention
- **Total Parameters**: 28,025,090

### Privacy Protection
- **Differential Privacy**: ε=8.0, δ=1e-5
- **Federated Learning**: Distributed training capability
- **Data Minimization**: Automated PHI removal

## Implementation

### Core Components

1. **Data Extraction (`src/data_processing/extract_mimic_data.py`)**
   - Identifies mental health patients using ICD codes
   - Extracts clinical notes, vital signs, medications
   - Creates temporal windows around mental health events

2. **Mental Health Phenotyping (`src/data_processing/mental_health_phenotyping.py`)**
   - Advanced NLP analysis of clinical notes
   - Medication pattern analysis
   - Crisis prediction window generation

3. **Self-Supervised Learning (`src/models/self_supervised_multimodal.py`)**
   - Multimodal fusion architecture
   - Contrastive learning framework
   - Privacy-preserving training

### Key Results

- **Mental Health Cohort**: 9,151 patients (19.7% of MIMIC-IV)
- **Clinical Data**: 88,418 relevant notes, 1.1M prescriptions
- **Model Training**: Successful privacy-preserving SSL implementation

## Project Structure

```
Assignment7-High Risk Project/
├── src/
│   ├── data_processing/
│   │   ├── extract_mimic_data.py
│   │   └── mental_health_phenotyping.py
│   └── models/
│       └── self_supervised_multimodal.py
├── configs/
│   └── project_config.py
├── notebooks/
│   └── exploration/
│       └── 01_data_exploration.ipynb
├── docs/
│   ├── AcademicReport.pdf
│   └── Presentation.pptx
├── results/
├── data/
├── requirements.txt
└── run_project.py
```

## Requirements

```bash
# Install dependencies
pip install -r requirements.txt

# Activate environment
conda activate msai
```

## Usage

1. **Complete Pipeline**:
   ```bash
   python run_project.py
   ```

2. **Individual Components**:
   ```bash
   # Data extraction
   python src/data_processing/extract_mimic_data.py
   
   # Mental health phenotyping
   python src/data_processing/mental_health_phenotyping.py
   
   # Model training
   python src/models/self_supervised_multimodal.py
   ```

3. **Data Exploration**:
   ```bash
   jupyter notebook notebooks/exploration/01_data_exploration.ipynb
   ```

## High-Risk Project Elements

### Technical Innovation
- Novel self-supervised learning approach for mental health
- Multimodal fusion of clinical data types
- Privacy-preserving training mechanisms
- Early prediction capability (2-4 weeks advance)

### Methodological Challenges
- Mental health crisis definition and validation
- Temporal modeling of long-range dependencies
- Class imbalance in crisis events
- Evaluation without ground truth

### Clinical Relevance
- Early warning system architecture
- Integration with clinical workflows
- Uncertainty quantification
- Human-in-the-loop design principles

### Ethical Framework
- Privacy protection mechanisms
- Bias mitigation strategies
- Transparent decision making
- Patient autonomy preservation

## Results and Evaluation

### Data Processing Results
- **Patient Identification**: Successfully identified 9,151 mental health patients
- **Data Extraction**: Processed 455,986 clinical notes
- **Feature Engineering**: Created multimodal temporal features
- **Privacy Compliance**: Implemented PHI removal and differential privacy

### Model Performance
- **Architecture Validation**: 28M parameter model successfully trained
- **Privacy Budget**: Maintained ε=8.0 differential privacy
- **Training Stability**: Convergent loss on synthetic data
- **Framework Completeness**: End-to-end pipeline operational

### Clinical Insights
- **Prevalence**: 19.7% mental health prevalence in MIMIC-IV
- **Data Richness**: Average 50 notes per mental health patient
- **Medication Complexity**: Average 7.8 psychiatric medications per patient
- **Temporal Patterns**: Crisis prediction windows successfully defined

## Limitations and Future Work

### Current Limitations
- Limited validation on real crisis events
- Synthetic data used for model testing
- Temporal calculation edge cases
- Computational resource requirements

### Future Directions
- Prospective clinical validation studies
- Integration with electronic health records
- Real-time prediction system deployment
- Regulatory approval pathway development

## Documentation

- **Final Report**: `docs/AcademicReport.pdf`
- **Presentation**: `docs/Presentation.pptx`
- **Configuration**: `configs/project_config.py`
- **Results**: `results/evaluation_summary.json`

## Technical Specifications

### Computing Environment
- **Python**: 3.12+
- **Framework**: PyTorch 2.6+
- **Privacy**: Opacus (differential privacy)
- **NLP**: Transformers (Clinical BERT)
- **Data**: Pandas, NumPy

### Privacy and Security
- **Differential Privacy**: Formal privacy guarantees
- **Data Minimization**: Automatic PHI removal
- **Audit Logging**: Complete data access tracking
- **Secure Processing**: Encrypted data handling

## Acknowledgments

This project builds upon significant contributions from the research community and open-source software ecosystem:

### Datasets and Medical Resources
- **MIMIC-IV Database**: Johnson et al. (2021) - Critical care database from Beth Israel Deaconess Medical Center
- **PhysioNet**: Goldberger et al. (2000) - Repository for physiological data
- **ICD-9/10 Classification**: World Health Organization - International disease classification standards

### Clinical NLP and Language Models
- **Bio_ClinicalBERT**: Alsentzer et al. (2019) - Pre-trained clinical language model
- **ScispaCy**: Neumann et al. (2019) - Biomedical text processing framework
- **BioBERT**: Lee et al. (2020) - Biomedical domain BERT model
- **Transformers Library**: Wolf et al. (2020) - State-of-the-art NLP framework

### Machine Learning Frameworks
- **PyTorch**: Paszke et al. (2019) - Deep learning framework
- **Scikit-learn**: Pedregosa et al. (2011) - Machine learning library
- **Self-Supervised Learning**: Chen et al. (2020) - Contrastive learning methodology

### Privacy-Preserving Technologies
- **Opacus**: Yousefpour et al. (2021) - Differential privacy for PyTorch
- **Differential Privacy Theory**: Dwork & Roth (2014) - Algorithmic foundations
- **Federated Learning**: Li et al. (2020) - Distributed privacy-preserving ML

### Mental Health Research
- **Clinical Prediction**: Rumshisky et al. (2016) - Psychiatric readmission prediction
- **EHR Phenotyping**: Castro et al. (2015) - Mental health condition validation
- **Healthcare AI Ethics**: Chen et al. (2021) - Ethical considerations in healthcare ML

### Technical Infrastructure
- **Time Series Analysis**: Christ et al. (2018) - tsfresh feature extraction
- **Multimodal Learning**: Radford et al. (2021) - Cross-modal representation learning
- **Healthcare Applications**: Rajkomar et al. (2019) - Machine learning in medicine

