#!/usr/bin/env python3
"""
MIMIC-IV Mental Health Data Extraction

This script extracts and preprocesses MIMIC-IV data to identify mental health patterns
and create datasets for self-supervised learning. note we also use synthetic data.

Key Mental Health ICD-9/10 Codes:
- Depression: 296.xx, F32-F33
- Anxiety: 300.xx, F40-F41  
- Delirium: 293.xx, F05
- Suicide attempts: E950-E959, X60-X84
- Bipolar: 296.xx, F31
- PTSD: 309.81, F43.1
- Substance abuse with mental health: 291-292, F10-F19

"""


import pandas as pd
import numpy as np
import sqlite3
import os
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Configuration
MIMIC_DATA_PATH = "/Users/dutts/Desktop/Saj/_Postgraduate/MSAI/2. AI in Healthcare/Assignments/data/mimic-iv"
PROJECT_DATA_PATH = "../data"
MENTAL_HEALTH_CODES = {
    'depression': ['296.2', '296.3', '311', 'F32', 'F33'],
    'anxiety': ['300.0', '300.2', '300.3', 'F40', 'F41'],
    'delirium': ['293.0', '293.1', 'F05'],
    'suicide_attempt': ['E950', 'E951', 'E952', 'E953', 'E954', 'E955', 'E956', 'E957', 'E958', 'E959',
                       'X60', 'X61', 'X62', 'X63', 'X64', 'X65', 'X66', 'X67', 'X68', 'X69',
                       'X70', 'X71', 'X72', 'X73', 'X74', 'X75', 'X76', 'X77', 'X78', 'X79',
                       'X80', 'X81', 'X82', 'X83', 'X84'],
    'bipolar': ['296.0', '296.1', '296.4', '296.5', '296.6', '296.7', 'F31'],
    'ptsd': ['309.81', 'F43.1'],
    'substance_mental': ['291', '292', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19']
}

class MIMICMentalHealthExtractor:
    """Extract and process MIMIC-IV data for mental health analysis."""
    
    def __init__(self, mimic_path: str, output_path: str):
        self.mimic_path = Path(mimic_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Track data statistics
        self.stats = {
            'total_patients': 0,
            'mental_health_patients': 0,
            'by_condition': {},
            'data_availability': {}
        }
        
    def load_compressed_csv(self, filename: str) -> pd.DataFrame:
        """Load a compressed CSV file from MIMIC-IV with error handling."""
        file_path = Path(MIMIC_DATA_PATH) / f"{filename}.csv.gz"
        print(f"Loading {filename}...")
        
        try:
            # Use chunking for large files to avoid memory issues
            if filename in ['NOTEEVENTS', 'CHARTEVENTS', 'PRESCRIPTIONS']:
                chunks = []
                chunk_size = 50000
                for chunk in pd.read_csv(file_path, compression='gzip', chunksize=chunk_size, low_memory=False):
                    chunks.append(chunk)
                    if len(chunks) * chunk_size > 500000:  # Limit to ~500k rows for memory
                        break
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(file_path, compression='gzip', low_memory=False)
            
            print(f"   Loaded {len(df):,} rows")
            return df
            
        except FileNotFoundError:
            print(f"   Error: File {file_path} not found")
            return pd.DataFrame()
        except Exception as e:
            print(f"   Error loading {filename}: {e}")
            return pd.DataFrame()
    
    def identify_mental_health_patients(self) -> pd.DataFrame:
        """Identify patients with mental health conditions using ICD codes."""
        print("\nIDENTIFYING MENTAL HEALTH PATIENTS")
        print("=" * 50)
        
        # Load diagnosis data
        diagnoses = self.load_compressed_csv('DIAGNOSES_ICD')
        
        # Create mental health flags
        mental_health_patients = []
        
        for condition, codes in MENTAL_HEALTH_CODES.items():
            condition_patients = set()
            
            for code in codes:
                # Match ICD codes (partial matching for code families)
                mask = diagnoses['ICD9_CODE'].astype(str).str.startswith(code, na=False)
                condition_patients.update(diagnoses[mask]['SUBJECT_ID'].unique())
            
            print(f"   {condition}: {len(condition_patients):,} patients")
            self.stats['by_condition'][condition] = len(condition_patients)
            
            # Add to mental health patient list
            for patient_id in condition_patients:
                mental_health_patients.append({
                    'SUBJECT_ID': patient_id,
                    'condition': condition,
                    'has_mental_health': True
                })
        
        # Convert to DataFrame
        mh_df = pd.DataFrame(mental_health_patients)
        unique_patients = mh_df['SUBJECT_ID'].nunique()
        
        self.stats['mental_health_patients'] = unique_patients
        print(f"\nTotal unique mental health patients: {unique_patients:,}")
        
        return mh_df
    
    def extract_patient_demographics(self, mental_health_patients: pd.DataFrame) -> pd.DataFrame:
        """Extract demographic information for mental health patients."""
        print("\nEXTRACTING PATIENT DEMOGRAPHICS")
        print("=" * 50)
        
        # Load patient data
        patients = self.load_compressed_csv('PATIENTS')
        
        # Merge with mental health patients
        mh_demographics = mental_health_patients.merge(
            patients[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'EXPIRE_FLAG']], 
            on='SUBJECT_ID', 
            how='left'
        )
        
        # Calculate age at first admission with proper error handling
        admissions = self.load_compressed_csv('ADMISSIONS')
        first_admission = admissions.groupby('SUBJECT_ID')['ADMITTIME'].min().reset_index()
        
        # Convert to datetime with error handling
        first_admission['ADMITTIME'] = pd.to_datetime(first_admission['ADMITTIME'], errors='coerce')
        
        mh_demographics = mh_demographics.merge(first_admission, on='SUBJECT_ID', how='left')
        
        # Convert DOB with error handling
        mh_demographics['DOB'] = pd.to_datetime(mh_demographics['DOB'], errors='coerce')
        
        # Calculate age with safe conversion
        try:
            age_diff = mh_demographics['ADMITTIME'] - mh_demographics['DOB']
            mh_demographics['age_at_admission'] = age_diff.dt.total_seconds() / (365.25 * 24 * 3600)
            # Cap unrealistic ages
            mh_demographics['age_at_admission'] = mh_demographics['age_at_admission'].clip(0, 120)
        except (OverflowError, ValueError) as e:
            print(f"   Warning: Age calculation error, using approximation: {e}")
            # Use year-based approximation as fallback
            mh_demographics['age_at_admission'] = (
                mh_demographics['ADMITTIME'].dt.year - mh_demographics['DOB'].dt.year
            ).clip(0, 120)
        
        print(f"   Demographics for {len(mh_demographics):,} patient records")
        
        return mh_demographics
    
    def extract_clinical_notes(self, mental_health_patients: pd.DataFrame) -> pd.DataFrame:
        """Extract clinical notes for mental health patients."""
        print("\n EXTRACTING CLINICAL NOTES")
        print("=" * 50)
        
        # Load note events
        notes = self.load_compressed_csv('NOTEEVENTS')
        
        # Filter for mental health patients
        mh_notes = notes[notes['SUBJECT_ID'].isin(mental_health_patients['SUBJECT_ID'])]
        
        # Focus on relevant note categories for mental health
        relevant_categories = [
            'Discharge summary',
            'Nursing',
            'Physician',
            'Social Work',
            'Case Management',
            'Consult',
            'Psychology',
            'Psychiatry'
        ]
        
        mh_notes_filtered = mh_notes[
            mh_notes['CATEGORY'].isin(relevant_categories)
        ].copy()
        
        # Clean and process text
        mh_notes_filtered['TEXT_LENGTH'] = mh_notes_filtered['TEXT'].astype(str).str.len()
        
        # Remove very short notes (likely empty or minimal)
        mh_notes_filtered = mh_notes_filtered[mh_notes_filtered['TEXT_LENGTH'] > 100]
        
        print(f"    Extracted {len(mh_notes_filtered):,} relevant clinical notes")
        print(f"    Note categories: {mh_notes_filtered['CATEGORY'].value_counts().to_dict()}")
        
        self.stats['data_availability']['clinical_notes'] = len(mh_notes_filtered)
        
        return mh_notes_filtered
    
    def extract_vital_signs(self, mental_health_patients: pd.DataFrame) -> pd.DataFrame:
        """Extract vital signs and physiological data."""
        print("\n EXTRACTING VITAL SIGNS")
        print("=" * 50)
        
        # Load chart events (this is a large file, so we'll sample)
        chartevents = self.load_compressed_csv('CHARTEVENTS')
        
        # Filter for mental health patients
        mh_charts = chartevents[chartevents['SUBJECT_ID'].isin(mental_health_patients['SUBJECT_ID'])]
        
        # Define vital signs of interest for mental health
        vital_itemids = {
            'heart_rate': [211, 220045],
            'systolic_bp': [51, 442, 455, 6701, 220179, 220050],
            'diastolic_bp': [8368, 8440, 8441, 8555, 220180, 220051],
            'respiratory_rate': [615, 618, 220210, 224690],
            'temperature': [223761, 678],
            'spo2': [646, 220277],
            'glasgow_coma': [198, 226755],
            'pain_score': [225908, 224842]
        }
        
        # Extract vital signs
        vital_data = []
        for vital_name, itemids in vital_itemids.items():
            vital_subset = mh_charts[mh_charts['ITEMID'].isin(itemids)].copy()
            vital_subset['vital_type'] = vital_name
            vital_data.append(vital_subset[['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'vital_type']])
        
        if vital_data:
            mh_vitals = pd.concat(vital_data, ignore_index=True)
            print(f"    Extracted {len(mh_vitals):,} vital sign measurements")
            print(f"    Vital types: {mh_vitals['vital_type'].value_counts().to_dict()}")
            
            self.stats['data_availability']['vital_signs'] = len(mh_vitals)
            return mh_vitals
        else:
            print("     No vital signs found")
            return pd.DataFrame()
    
    def extract_medications(self, mental_health_patients: pd.DataFrame) -> pd.DataFrame:
        """Extract medication data, focusing on psychiatric medications."""
        print("\n EXTRACTING MEDICATIONS")
        print("=" * 50)
        
        # Load prescriptions
        prescriptions = self.load_compressed_csv('PRESCRIPTIONS')
        
        # Filter for mental health patients
        mh_meds = prescriptions[prescriptions['SUBJECT_ID'].isin(mental_health_patients['SUBJECT_ID'])]
        
        # Psychiatric medication keywords
        psych_keywords = [
            'sertraline', 'fluoxetine', 'paroxetine', 'citalopram', 'escitalopram',  # SSRIs
            'venlafaxine', 'duloxetine', 'desvenlafaxine',  # SNRIs
            'lithium', 'valproate', 'lamotrigine', 'carbamazepine',  # Mood stabilizers
            'risperidone', 'olanzapine', 'quetiapine', 'aripiprazole', 'haloperidol',  # Antipsychotics
            'lorazepam', 'clonazepam', 'alprazolam', 'diazepam',  # Anxiolytics
            'trazodone', 'mirtazapine', 'bupropion'  # Other antidepressants
        ]
        
        # Find psychiatric medications
        psych_mask = mh_meds['DRUG'].str.lower().str.contains('|'.join(psych_keywords), na=False)
        mh_psych_meds = mh_meds[psych_mask].copy()
        
        print(f"    Found {len(mh_psych_meds):,} psychiatric medication prescriptions")
        print(f"    Top medications: {mh_psych_meds['DRUG'].value_counts().head().to_dict()}")
        
        self.stats['data_availability']['medications'] = len(mh_psych_meds)
        
        return mh_psych_meds
    
    def extract_lab_results(self, mental_health_patients: pd.DataFrame) -> pd.DataFrame:
        """Extract laboratory results relevant to mental health."""
        print("\n EXTRACTING LABORATORY RESULTS")
        print("=" * 50)
        
        # Load lab events
        labevents = self.load_compressed_csv('LABEVENTS')
        
        # Filter for mental health patients
        mh_labs = labevents[labevents['SUBJECT_ID'].isin(mental_health_patients['SUBJECT_ID'])]
        
        # Mental health relevant lab tests
        # Note: This is a simplified list - in practice, you'd need to map ITEMID to specific tests
        print(f"    Found {len(mh_labs):,} lab results for mental health patients")
        
        self.stats['data_availability']['lab_results'] = len(mh_labs)
        
        return mh_labs
    
    def create_temporal_windows(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Create temporal windows around mental health events for self-supervised learning."""
        print("\n CREATING TEMPORAL WINDOWS")
        print("=" * 50)
        
        # This is where we would implement the complex temporal windowing logic
        # For now, we'll create a placeholder structure
        
        temporal_data = {}
        
        # Example: Create 2-week windows before and after mental health diagnoses
        for data_type, df in data_dict.items():
            if not df.empty and 'CHARTTIME' in df.columns:
                # Sort by time
                df_sorted = df.sort_values(['SUBJECT_ID', 'CHARTTIME'])
                temporal_data[f"{data_type}_temporal"] = df_sorted
                print(f"    Created temporal windows for {data_type}")
        
        return temporal_data
    
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics."""
        print("\n SUMMARY STATISTICS")
        print("=" * 50)
        
        for key, value in self.stats.items():
            if isinstance(value, dict):
                print(f"\n{key.upper()}:")
                for subkey, subvalue in value.items():
                    print(f"   {subkey}: {subvalue:,}")
            else:
                print(f"{key}: {value:,}")
    
    def save_processed_data(self, data_dict: Dict[str, pd.DataFrame]):
        """Save all processed data to the output directory."""
        print("\n SAVING PROCESSED DATA")
        print("=" * 50)
        
        for name, df in data_dict.items():
            if not df.empty:
                output_file = self.output_path / f"{name}.csv"
                df.to_csv(output_file, index=False)
                print(f"    Saved {name}: {len(df):,} rows → {output_file}")
        
        # Save statistics
        stats_file = self.output_path / "extraction_statistics.txt"
        with open(stats_file, 'w') as f:
            f.write("MIMIC-IV Mental Health Data Extraction Statistics\n")
            f.write("=" * 50 + "\n\n")
            for key, value in self.stats.items():
                if isinstance(value, dict):
                    f.write(f"{key.upper()}:\n")
                    for subkey, subvalue in value.items():
                        f.write(f"   {subkey}: {subvalue:,}\n")
                    f.write("\n")
                else:
                    f.write(f"{key}: {value:,}\n")
        
        print(f"    Statistics saved to {stats_file}")

def main():
    """Main execution function with comprehensive error handling."""
    print("MIMIC-IV MENTAL HEALTH DATA EXTRACTION")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize extractor
    extractor = MIMICMentalHealthExtractor(MIMIC_DATA_PATH, PROJECT_DATA_PATH)
    
    try:
        # Step 1: Identify mental health patients
        print("Step 1: Identifying mental health patients...")
        mental_health_patients = extractor.identify_mental_health_patients()
        
        if mental_health_patients.empty:
            print("No mental health patients found. Check data path and files.")
            return
        
        # Step 2: Extract demographics
        print("Step 2: Extracting patient demographics...")
        try:
            demographics = extractor.extract_patient_demographics(mental_health_patients)
        except Exception as e:
            print(f"Warning: Demographics extraction failed: {e}")
            demographics = mental_health_patients.copy()  # Use basic patient data
        
        # Step 3: Extract multimodal data
        print("Step 3: Extracting multimodal data...")
        try:
            clinical_notes = extractor.extract_clinical_notes(mental_health_patients)
        except Exception as e:
            print(f"Warning: Clinical notes extraction failed: {e}")
            clinical_notes = pd.DataFrame()
        
        try:
            vital_signs = extractor.extract_vital_signs(mental_health_patients)
        except Exception as e:
            print(f"Warning: Vital signs extraction failed: {e}")
            vital_signs = pd.DataFrame()
        
        try:
            medications = extractor.extract_medications(mental_health_patients)
        except Exception as e:
            print(f"Warning: Medications extraction failed: {e}")
            medications = pd.DataFrame()
        
        try:
            lab_results = extractor.extract_lab_results(mental_health_patients)
        except Exception as e:
            print(f"Warning: Lab results extraction failed: {e}")
            lab_results = pd.DataFrame()
        
        # Step 4: Create temporal windows
        print("Step 4: Creating temporal windows...")
        data_dict = {
            'mental_health_patients': mental_health_patients,
            'demographics': demographics,
            'clinical_notes': clinical_notes,
            'vital_signs': vital_signs,
            'medications': medications,
            'lab_results': lab_results
        }
        
        try:
            temporal_data = extractor.create_temporal_windows(data_dict)
            data_dict.update(temporal_data)
        except Exception as e:
            print(f"Warning: Temporal window creation failed: {e}")
        
        # Step 5: Save processed data
        extractor.save_processed_data(data_dict)
        
        # Step 6: Generate summary
        extractor.generate_summary_statistics()

        print(f" Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n ERROR OCCURRED: {str(e)}")
        raise

if __name__ == "__main__":
    main()
