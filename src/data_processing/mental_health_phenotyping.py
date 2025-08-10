#!/usr/bin/env python3
"""
Mental Health Phenotyping and Crisis Detection
-

This script builds on existing assignment work from Assignment 1 to 5 and all learninings 
to create sophisticated mental health phenotypes and identify crisis patterns in MIMIC-IV data.

Leverages specifically:
- Assignment 3: NLP processing for clinical notes
- Assignment 5: ML pipelines for feature engineering  
- Assignment 6: LLM reasoning for clinical insights

"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Dict, List, Tuple, Optional
import logging


warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MentalHealthPhenotyper:
    """
    Advanced mental health phenotyping using multimodal clinical data.
    
    This class identifies mental health crisis patterns by:
    1. Analyzing clinical notes for mental health indicators
    2. Tracking medication patterns and changes
    3. Monitoring physiological crisis markers
    4. Creating temporal crisis prediction windows
    """
    
    def __init__(self, mimic_path: str, output_path: str):
        self.mimic_path = Path(mimic_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Mental health terminology from clinical literature
        self.mental_health_keywords = {
            'depression_indicators': [
                'depressed', 'depression', 'sad', 'hopeless', 'worthless',
                'anhedonia', 'dysthymia', 'melancholy', 'despair', 'suicidal ideation'
            ],
            'anxiety_indicators': [
                'anxious', 'anxiety', 'panic', 'worried', 'restless', 'agitated',
                'nervous', 'apprehensive', 'fearful', 'panic attack'
            ],
            'psychosis_indicators': [
                'hallucination', 'delusion', 'paranoid', 'psychotic', 'disorganized',
                'thought disorder', 'bizarre behavior', 'hearing voices'
            ],
            'delirium_indicators': [
                'confused', 'disoriented', 'altered mental status', 'agitated',
                'combative', 'delirious', 'sundowning', 'acute confusion'
            ],
            'crisis_indicators': [
                'suicide', 'self-harm', 'overdose', 'crisis', 'emergency',
                'psychiatric emergency', 'behavioral emergency', 'sitter required',
                '1:1 observation', 'psychiatric hold'
            ],
            'substance_indicators': [
                'withdrawal', 'intoxication', 'substance abuse', 'addiction',
                'alcohol abuse', 'drug abuse', 'detox', 'rehabilitation'
            ]
        }
        
        # Crisis prediction time windows
        self.crisis_windows = {
            'immediate': 1,      # 1 day - immediate crisis
            'short_term': 7,     # 1 week - short-term risk
            'medium_term': 14,   # 2 weeks - medium-term prediction
            'long_term': 28      # 4 weeks - early prediction target
        }
        
        # Psychiatric medications for tracking
        self.psych_medications = {
            'antidepressants': {
                'ssri': ['sertraline', 'fluoxetine', 'paroxetine', 'citalopram', 'escitalopram'],
                'snri': ['venlafaxine', 'duloxetine', 'desvenlafaxine'],
                'tricyclic': ['amitriptyline', 'nortriptyline', 'imipramine'],
                'other': ['bupropion', 'mirtazapine', 'trazodone']
            },
            'anxiolytics': ['lorazepam', 'clonazepam', 'alprazolam', 'diazepam', 'midazolam'],
            'antipsychotics': {
                'typical': ['haloperidol', 'chlorpromazine', 'fluphenazine'],
                'atypical': ['risperidone', 'olanzapine', 'quetiapine', 'aripiprazole', 'ziprasidone']
            },
            'mood_stabilizers': ['lithium', 'valproate', 'lamotrigine', 'carbamazepine'],
            'stimulants': ['methylphenidate', 'amphetamine', 'dextroamphetamine']
        }
    
    def load_mimic_table(self, table_name: str) -> pd.DataFrame:
        """Load MIMIC-IV table with error handling."""
        filepath = self.mimic_path / f"{table_name}.csv.gz"
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return pd.DataFrame()
        
        try:
            logger.info(f"Loading {table_name}...")
            df = pd.read_csv(filepath, compression='gzip', low_memory=False)
            logger.info(f"Loaded {len(df):,} rows from {table_name}")
            return df
        except Exception as e:
            logger.error(f"Error loading {table_name}: {e}")
            return pd.DataFrame()
    
    def identify_mental_health_patients(self) -> pd.DataFrame:
        """Identify patients with mental health conditions using comprehensive approach."""
        
        logger.info(" IDENTIFYING MENTAL HEALTH PATIENTS")
        logger.info("=" * 50)
        
        # Load diagnosis data
        diagnoses = self.load_mimic_table('DIAGNOSES_ICD')
        if diagnoses.empty:
            return pd.DataFrame()
        
        # Mental health ICD codes (expanded from literature)
        mental_health_icd = {
            'mood_disorders': [
                # Major Depression
                '296.2', '296.3', '311', 'F32', 'F33',
                # Bipolar
                '296.0', '296.1', '296.4', '296.5', '296.6', '296.7', 'F31',
                # Dysthymia
                '300.4', 'F34.1'
            ],
            'anxiety_disorders': [
                # Generalized Anxiety
                '300.02', 'F41.1',
                # Panic Disorder
                '300.01', 'F41.0',
                # Specific Phobias
                '300.29', 'F40',
                # PTSD
                '309.81', 'F43.1',
                # OCD
                '300.3', 'F42'
            ],
            'psychotic_disorders': [
                # Schizophrenia
                '295', 'F20',
                # Brief Psychotic Disorder
                '298.8', 'F23',
                # Delusional Disorder
                '297.1', 'F22'
            ],
            'delirium_cognitive': [
                # Delirium
                '293.0', '293.1', 'F05',
                # Dementia
                '290', 'F03', 'F01', 'F02'
            ],
            'substance_mental': [
                # Alcohol-related
                '291', '303', 'F10',
                # Drug-related
                '292', '304', '305', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19'
            ],
            'crisis_codes': [
                # Suicide attempts
                'E950', 'E951', 'E952', 'E953', 'E954', 'E955', 'E956', 'E957', 'E958', 'E959',
                'X60', 'X61', 'X62', 'X63', 'X64', 'X65', 'X66', 'X67', 'X68', 'X69',
                'X70', 'X71', 'X72', 'X73', 'X74', 'X75', 'X76', 'X77', 'X78', 'X79',
                'X80', 'X81', 'X82', 'X83', 'X84',
                # Self-harm
                'T14.91', 'Z91.5'
            ]
        }
        
        # Find patients for each category
        mental_health_patients = []
        stats = {}
        
        for category, codes in mental_health_icd.items():
            category_patients = set()
            
            for code in codes:
                # Handle both ICD-9 and ICD-10
                mask = diagnoses['ICD9_CODE'].astype(str).str.startswith(code, na=False)
                matching_patients = diagnoses[mask]['SUBJECT_ID'].unique()
                category_patients.update(matching_patients)
            
            stats[category] = len(category_patients)
            logger.info(f"   {category:<20}: {len(category_patients):>6,} patients")
            
            # Add to main list
            for patient_id in category_patients:
                mental_health_patients.append({
                    'SUBJECT_ID': patient_id,
                    'condition_category': category,
                    'has_mental_health': True
                })
        
        # Convert to DataFrame
        mh_df = pd.DataFrame(mental_health_patients)
        
        if not mh_df.empty:
            unique_patients = mh_df['SUBJECT_ID'].nunique()
            total_patients = diagnoses['SUBJECT_ID'].nunique()
            
            logger.info(f"\\n Summary:")
            logger.info(f"   Total MIMIC-IV patients: {total_patients:,}")
            logger.info(f"   Mental health patients: {unique_patients:,}")
            logger.info(f"   Prevalence: {(unique_patients/total_patients)*100:.1f}%")
        
        return mh_df
    
    def extract_clinical_notes_with_nlp(self, mental_health_patients: pd.DataFrame) -> pd.DataFrame:
        """Extract and process clinical notes using NLP (adapted from Assignment 3)."""
        
        logger.info(" PROCESSING CLINICAL NOTES WITH NLP")
        logger.info("=" * 50)
        
        # Load note events
        notes = self.load_mimic_table('NOTEEVENTS')
        if notes.empty:
            return pd.DataFrame()
        
        # Filter for mental health patients
        mh_patient_ids = mental_health_patients['SUBJECT_ID'].unique()
        mh_notes = notes[notes['SUBJECT_ID'].isin(mh_patient_ids)].copy()
        
        logger.info(f"Found {len(mh_notes):,} notes for {len(mh_patient_ids):,} mental health patients")
        
        # Focus on relevant note categories
        relevant_categories = [
            'Discharge summary', 'Nursing', 'Physician', 'Social Work',
            'Case Management', 'Consult', 'Psychology', 'Psychiatry'
        ]
        
        mh_notes_filtered = mh_notes[mh_notes['CATEGORY'].isin(relevant_categories)].copy()
        
        # Clean text data
        mh_notes_filtered['TEXT_CLEAN'] = mh_notes_filtered['TEXT'].astype(str)
        mh_notes_filtered['TEXT_LENGTH'] = mh_notes_filtered['TEXT_CLEAN'].str.len()
        
        # Remove very short notes
        mh_notes_filtered = mh_notes_filtered[mh_notes_filtered['TEXT_LENGTH'] > 100]
        
        logger.info(f"After filtering: {len(mh_notes_filtered):,} relevant notes")
        
        # Apply NLP processing (adapted from Assignment 3 approach)
        mh_notes_processed = self.process_notes_for_mental_health(mh_notes_filtered)
        
        return mh_notes_processed
    
    def process_notes_for_mental_health(self, notes: pd.DataFrame) -> pd.DataFrame:
        """Process clinical notes for mental health indicators."""
        
        logger.info(" Analyzing notes for mental health indicators...")
        
        notes_processed = notes.copy()
        
        # Score each note for mental health content
        for category, keywords in self.mental_health_keywords.items():
            # Create keyword pattern
            pattern = '|'.join([re.escape(keyword) for keyword in keywords])
            
            # Count matches (case insensitive)
            notes_processed[f'{category}_count'] = notes_processed['TEXT_CLEAN'].str.count(
                pattern, flags=re.IGNORECASE
            )
            
            # Binary indicator
            notes_processed[f'has_{category}'] = (notes_processed[f'{category}_count'] > 0).astype(int)
        
        # Calculate overall mental health severity score
        indicator_columns = [col for col in notes_processed.columns if col.startswith('has_')]
        notes_processed['mental_health_severity'] = notes_processed[indicator_columns].sum(axis=1)
        
        # Add temporal features
        notes_processed['CHARTDATE'] = pd.to_datetime(notes_processed['CHARTDATE'])
        notes_processed['hour_of_day'] = notes_processed['CHARTDATE'].dt.hour
        notes_processed['day_of_week'] = notes_processed['CHARTDATE'].dt.dayofweek
        
        logger.info(f"Processed notes with mental health scoring")
        logger.info(f"Average mental health severity: {notes_processed['mental_health_severity'].mean():.2f}")
        
        return notes_processed
    
    def analyze_medication_patterns(self, mental_health_patients: pd.DataFrame) -> pd.DataFrame:
        """Analyze psychiatric medication patterns for crisis prediction."""
        
        logger.info(" ANALYZING MEDICATION PATTERNS")
        logger.info("=" * 50)
        
        # Load prescriptions
        prescriptions = self.load_mimic_table('PRESCRIPTIONS')
        if prescriptions.empty:
            return pd.DataFrame()
        
        # Filter for mental health patients
        mh_patient_ids = mental_health_patients['SUBJECT_ID'].unique()
        mh_prescriptions = prescriptions[prescriptions['SUBJECT_ID'].isin(mh_patient_ids)].copy()
        
        logger.info(f"Found {len(mh_prescriptions):,} prescriptions for mental health patients")
        
        # Identify psychiatric medications
        mh_prescriptions['drug_lower'] = mh_prescriptions['DRUG'].str.lower()
        
        # Categorize medications
        for med_category, med_subcategories in self.psych_medications.items():
            if isinstance(med_subcategories, dict):
                # Handle subcategories
                category_mask = pd.Series(False, index=mh_prescriptions.index)
                for subcat, drugs in med_subcategories.items():
                    for drug in drugs:
                        mask = mh_prescriptions['drug_lower'].str.contains(drug, na=False)
                        category_mask |= mask
                        mh_prescriptions.loc[mask, f'{med_category}_subtype'] = subcat
                
                mh_prescriptions[f'is_{med_category}'] = category_mask
            else:
                # Handle simple list
                category_mask = pd.Series(False, index=mh_prescriptions.index)
                for drug in med_subcategories:
                    mask = mh_prescriptions['drug_lower'].str.contains(drug, na=False)
                    category_mask |= mask
                
                mh_prescriptions[f'is_{med_category}'] = category_mask
        
        # Calculate medication complexity and changes
        mh_prescriptions_analyzed = self.analyze_medication_changes(mh_prescriptions)
        
        return mh_prescriptions_analyzed
    
    def analyze_medication_changes(self, prescriptions: pd.DataFrame) -> pd.DataFrame:
        """Analyze medication changes as crisis indicators."""
        
        logger.info(" Analyzing medication changes...")
        
        # Convert dates
        prescriptions['STARTDATE'] = pd.to_datetime(prescriptions['STARTDATE'])
        prescriptions['ENDDATE'] = pd.to_datetime(prescriptions['ENDDATE'])
        
        # Sort by patient and date
        prescriptions = prescriptions.sort_values(['SUBJECT_ID', 'STARTDATE'])
        
        # Calculate medication features per patient
        patient_med_features = []
        
        for patient_id in prescriptions['SUBJECT_ID'].unique():
            patient_meds = prescriptions[prescriptions['SUBJECT_ID'] == patient_id]
            
            # Count psychiatric medication categories
            psych_categories = [col for col in patient_meds.columns if col.startswith('is_')]
            psych_med_counts = patient_meds[psych_categories].sum()
            
            # Calculate medication complexity (number of different psych meds)
            complexity = psych_med_counts.sum()
            
            # Identify rapid medication changes (potential crisis indicator)
            patient_meds['days_between_changes'] = patient_meds['STARTDATE'].diff().dt.days
            rapid_changes = (patient_meds['days_between_changes'] < 7).sum()
            
            # Calculate total duration of psychiatric medications
            total_duration = (patient_meds['ENDDATE'] - patient_meds['STARTDATE']).dt.days.sum()
            
            patient_features = {
                'SUBJECT_ID': patient_id,
                'medication_complexity': complexity,
                'rapid_medication_changes': rapid_changes,
                'total_psych_med_duration': total_duration,
                'num_prescriptions': len(patient_meds)
            }
            
            # Add individual medication category counts
            for category in psych_categories:
                patient_features[f'count_{category}'] = psych_med_counts[category]
            
            patient_med_features.append(patient_features)
        
        medication_features_df = pd.DataFrame(patient_med_features)
        
        logger.info(f"Created medication features for {len(medication_features_df)} patients")
        logger.info(f"Average medication complexity: {medication_features_df['medication_complexity'].mean():.1f}")
        
        return medication_features_df
    
    def create_crisis_prediction_windows(self, 
                                       notes_processed: pd.DataFrame,
                                       medication_features: pd.DataFrame) -> pd.DataFrame:
        """Create temporal windows for crisis prediction."""
        
        logger.info(" CREATING CRISIS PREDICTION WINDOWS")
        logger.info("=" * 50)
        
        # Identify potential crisis events from notes
        crisis_indicators = ['crisis_indicators', 'has_crisis_indicators']
        
        crisis_events = []
        
        for _, note in notes_processed.iterrows():
            # Check if this note indicates a crisis
            is_crisis = False
            crisis_severity = 0
            
            for indicator in crisis_indicators:
                if indicator in note and note[indicator] > 0:
                    is_crisis = True
                    crisis_severity += note[indicator]
            
            if is_crisis:
                crisis_events.append({
                    'SUBJECT_ID': note['SUBJECT_ID'],
                    'crisis_date': note['CHARTDATE'],
                    'crisis_severity': crisis_severity,
                    'note_category': note['CATEGORY']
                })
        
        crisis_events_df = pd.DataFrame(crisis_events)
        
        if crisis_events_df.empty:
            logger.warning("No crisis events identified")
            return pd.DataFrame()
    
        crisis_events_df['crisis_date'] = pd.to_datetime(crisis_events_df['crisis_date'])
        
        logger.info(f"Identified {len(crisis_events_df)} potential crisis events")
        
        prediction_windows = []
        
        for _, crisis in crisis_events_df.iterrows():
            crisis_date = pd.to_datetime(crisis['crisis_date'])
            patient_id = crisis['SUBJECT_ID']
            
            for window_name, days_before in self.crisis_windows.items():
                window_start = crisis_date - timedelta(days=days_before)
                
                # Get notes in this window
                patient_notes = notes_processed[
                    (notes_processed['SUBJECT_ID'] == patient_id) &
                    (notes_processed['CHARTDATE'] >= window_start) &
                    (notes_processed['CHARTDATE'] < crisis_date)
                ]
                
                if len(patient_notes) > 0:
                    # Aggregate features for this window
                    window_features = {
                        'SUBJECT_ID': patient_id,
                        'crisis_date': crisis_date,
                        'window_type': window_name,
                        'window_start': window_start,
                        'days_before_crisis': days_before,
                        'num_notes_in_window': len(patient_notes),
                        'avg_mental_health_severity': patient_notes['mental_health_severity'].mean(),
                        'max_mental_health_severity': patient_notes['mental_health_severity'].max(),
                        'crisis_target': 1  # This window precedes a crisis
                    }
                    
                    # Add aggregated mental health indicators
                    for category in self.mental_health_keywords.keys():
                        count_col = f'{category}_count'
                        if count_col in patient_notes.columns:
                            window_features[f'total_{category}'] = patient_notes[count_col].sum()
                            window_features[f'avg_{category}'] = patient_notes[count_col].mean()
                    
                    prediction_windows.append(window_features)
        
        prediction_windows_df = pd.DataFrame(prediction_windows)
        
        # Add negative examples (windows not preceding crises)
        prediction_windows_with_negatives = self.add_negative_examples(
            prediction_windows_df, notes_processed
        )
        
        logger.info(f"Created {len(prediction_windows_with_negatives)} prediction windows")
        logger.info(f"   Positive examples (pre-crisis): {(prediction_windows_with_negatives['crisis_target'] == 1).sum()}")
        logger.info(f"   Negative examples (normal): {(prediction_windows_with_negatives['crisis_target'] == 0).sum()}")
        
        return prediction_windows_with_negatives
    
    def add_negative_examples(self, positive_windows: pd.DataFrame, 
                            notes_processed: pd.DataFrame) -> pd.DataFrame:
        """Add negative examples (non-crisis windows) for balanced training."""
        
        # Sample random windows that don't precede crises
        negative_windows = []
        
        # Get patients with positive examples
        crisis_patients = positive_windows['SUBJECT_ID'].unique()
        
        for patient_id in crisis_patients:
            patient_notes = notes_processed[notes_processed['SUBJECT_ID'] == patient_id]
            
            if len(patient_notes) < 10:  # Skip patients with too few notes
                continue
            
            # Sample random dates for negative windows
            note_dates = patient_notes['CHARTDATE'].sort_values()
            
            # Create negative windows (not within 30 days of any crisis)
            crisis_dates = positive_windows[positive_windows['SUBJECT_ID'] == patient_id]['crisis_date']
            
            for window_name, days_before in self.crisis_windows.items():
                # Sample a random date
                random_date = pd.to_datetime(np.random.choice(note_dates))
                
                # Check if this date is far from any crisis
                min_distance_to_crisis = float('inf')
                for crisis_date in crisis_dates:
                    crisis_date = pd.to_datetime(crisis_date)
                    distance = abs((random_date - crisis_date).days)
                    min_distance_to_crisis = min(min_distance_to_crisis, distance)
                
                # Only use if far from any crisis
                if min_distance_to_crisis > 30:
                    window_start = random_date - timedelta(days=days_before)
                    
                    # Get notes in this window
                    window_notes = patient_notes[
                        (patient_notes['CHARTDATE'] >= window_start) &
                        (patient_notes['CHARTDATE'] < random_date)
                    ]
                    
                    if len(window_notes) > 0:
                        window_features = {
                            'SUBJECT_ID': patient_id,
                            'crisis_date': None,
                            'window_type': window_name,
                            'window_start': window_start,
                            'days_before_crisis': days_before,
                            'num_notes_in_window': len(window_notes),
                            'avg_mental_health_severity': window_notes['mental_health_severity'].mean(),
                            'max_mental_health_severity': window_notes['mental_health_severity'].max(),
                            'crisis_target': 0  # No crisis follows
                        }
                        
                        # Add aggregated mental health indicators
                        for category in self.mental_health_keywords.keys():
                            count_col = f'{category}_count'
                            if count_col in window_notes.columns:
                                window_features[f'total_{category}'] = window_notes[count_col].sum()
                                window_features[f'avg_{category}'] = window_notes[count_col].mean()
                        
                        negative_windows.append(window_features)
        
        # Combine positive and negative windows
        negative_windows_df = pd.DataFrame(negative_windows)
        combined_windows = pd.concat([positive_windows, negative_windows_df], ignore_index=True)
        
        return combined_windows
    
    def generate_comprehensive_summary(self, all_data: Dict[str, pd.DataFrame]):
        """Generate comprehensive analysis summary."""
        
        logger.info(" GENERATING COMPREHENSIVE SUMMARY")
        logger.info("=" * 50)
        
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'data_summary': {},
            'mental_health_patterns': {},
            'crisis_prediction_readiness': {}
        }
        
        # Data summary
        for data_type, df in all_data.items():
            if not df.empty:
                summary['data_summary'][data_type] = {
                    'num_records': len(df),
                    'num_patients': df['SUBJECT_ID'].nunique() if 'SUBJECT_ID' in df.columns else 'N/A',
                    'columns': list(df.columns)
                }
        
        # Mental health patterns
        if 'prediction_windows' in all_data and not all_data['prediction_windows'].empty:
            windows_df = all_data['prediction_windows']
            
            summary['mental_health_patterns'] = {
                'total_prediction_windows': len(windows_df),
                'crisis_windows': (windows_df['crisis_target'] == 1).sum(),
                'normal_windows': (windows_df['crisis_target'] == 0).sum(),
                'class_balance': (windows_df['crisis_target'] == 1).mean(),
                'avg_mental_health_severity': windows_df['avg_mental_health_severity'].mean(),
                'window_types': windows_df['window_type'].value_counts().to_dict()
            }
        
        # Crisis prediction readiness
        if 'prediction_windows' in all_data:
            windows_df = all_data['prediction_windows']
            crisis_windows = windows_df[windows_df['crisis_target'] == 1]
            
            if len(crisis_windows) > 0:
                summary['crisis_prediction_readiness'] = {
                    'early_detection_feasible': len(crisis_windows[crisis_windows['days_before_crisis'] >= 14]) > 0,
                    'num_4_week_early_windows': len(crisis_windows[crisis_windows['days_before_crisis'] >= 28]),
                    'num_2_week_early_windows': len(crisis_windows[crisis_windows['days_before_crisis'] >= 14]),
                    'min_days_early_detection': crisis_windows['days_before_crisis'].max(),
                    'feature_completeness': {
                        'clinical_notes': 'avg_mental_health_severity' in windows_df.columns,
                        'medication_data': any('medication' in col for col in windows_df.columns),
                        'temporal_features': 'days_before_crisis' in windows_df.columns
                    }
                }
        
        # Save summary
        import json
        summary_file = self.output_path / "mental_health_phenotyping_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Comprehensive summary saved to {summary_file}")
        
        # Print key findings
        logger.info("\\n KEY FINDINGS:")
        logger.info(f"   Mental health patients identified: {summary['data_summary'].get('mental_health_patients', {}).get('num_patients', 'N/A')}")
        
        if 'mental_health_patterns' in summary:
            patterns = summary['mental_health_patterns']
            logger.info(f"   Crisis prediction windows: {patterns.get('crisis_windows', 0)}")
            logger.info(f"   Class balance: {patterns.get('class_balance', 0):.1%}")
        
        if 'crisis_prediction_readiness' in summary:
            readiness = summary['crisis_prediction_readiness']
            logger.info(f"   Early detection feasible: {readiness.get('early_detection_feasible', False)}")
            logger.info(f"   4-week early detection windows: {readiness.get('num_4_week_early_windows', 0)}")
        
        return summary

def main():
    """Main execution function."""
    
    print(" MENTAL HEALTH PHENOTYPING AND CRISIS DETECTION")
    print("=" * 60)
    print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    mimic_path = "/Users/dutts/Desktop/Saj/_Postgraduate/MSAI/2. AI in Healthcare/Assignments/data/mimic-iv"
    output_path = "../../data/processed"
    
    # Initialize phenotyper
    phenotyper = MentalHealthPhenotyper(mimic_path, output_path)
    
    try:
        # Step 1: Identify mental health patients
        mental_health_patients = phenotyper.identify_mental_health_patients()
        
        if mental_health_patients.empty:
            logger.error("No mental health patients identified. Check data paths and ICD codes.")
            return
        
        # Step 2: Process clinical notes with NLP
        notes_processed = phenotyper.extract_clinical_notes_with_nlp(mental_health_patients)
        
        # Step 3: Analyze medication patterns
        medication_features = phenotyper.analyze_medication_patterns(mental_health_patients)
        
        # Step 4: Create crisis prediction windows
        if not notes_processed.empty:
            prediction_windows = phenotyper.create_crisis_prediction_windows(
                notes_processed, medication_features
            )
        else:
            prediction_windows = pd.DataFrame()
        
        # Step 5: Compile all data
        all_data = {
            'mental_health_patients': mental_health_patients,
            'notes_processed': notes_processed,
            'medication_features': medication_features,
            'prediction_windows': prediction_windows
        }
        
        # Step 6: Save processed data
        for name, df in all_data.items():
            if not df.empty:
                output_file = Path(output_path) / f"{name}.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Saved {name}: {len(df):,} records → {output_file}")
        
        # Step 7: Generate comprehensive summary
        summary = phenotyper.generate_comprehensive_summary(all_data)
        
        print(f"\\n PHENOTYPING COMPLETED SUCCESSFULLY")
        print(f" Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
