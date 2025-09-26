import os
import pydicom
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import pytesseract
import pandas as pd
from typing import Dict, Any, Optional, List
import json
import re
import argparse
import sys


def extract_perfusion_parameters(folder_path: str,
                               output_format: str = 'dict',
                               save_to_file: bool = False,
                               output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract specific perfusion parameters (CBF, Tmax, CBV with thresholds and volumes) from DICOM images.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing DICOM files
    output_format : str
        Format for output data ('dict', 'dataframe', 'json')
    save_to_file : bool
        Whether to save results to a file
    output_file : str, optional
        Path for output file. If file exists, only new files will be processed and results appended.
    
    Returns:
    --------
    Dict[str, Any] or pd.DataFrame
        Extracted perfusion parameters including PatientID, file information, and measurements
        
    Notes:
    ------
    When output_format='dataframe', the result includes columns:
    - patient_id: DICOM PatientID field
    - file_path: path to the DICOM file  
    - acquisition_date: DICOM acquisition/study/series date
    - parameter_type: 'CBF', 'Tmax', or 'CBV'
    - threshold: parameter threshold (e.g., '<20%', '>6s', '<2.0ml/100g')
    - volume: measured volume value
    - unit: measurement unit
    
    If an existing output file is provided, only files not already processed will be analyzed,
    and results will be appended to the existing file.
    """
    
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path does not exist: {folder_path}")
    
    # Check for existing output file and read already processed files
    existing_data = pd.DataFrame()
    processed_files = set()
    processed_combinations = set()
    
    if save_to_file and output_file and os.path.exists(output_file):
        try:
            if output_file.endswith('.csv'):
                existing_data = pd.read_csv(output_file)
            elif output_file.endswith('.xlsx'):
                existing_data = pd.read_excel(output_file)
            
            if not existing_data.empty:
                # Create set of processed file paths (for backward compatibility)
                if 'file_path' in existing_data.columns:
                    processed_files = set(existing_data['file_path'].unique())
                
                # Create set of processed combinations (patient_id, acquisition_date, file_path)
                # This is more robust for detecting duplicates
                if all(col in existing_data.columns for col in ['patient_id', 'acquisition_date', 'file_path']):
                    # Normalize the existing data first to ensure consistent string conversion
                    normalized_existing = _normalize_dataframe_types(existing_data)
                    for _, row in normalized_existing.iterrows():
                        combo = (str(row['patient_id']), str(row['acquisition_date']), str(row['file_path']))
                        processed_combinations.add(combo)
                
                print(f"Found existing output file with {len(processed_files)} already processed files")
                print(f"Tracking {len(processed_combinations)} unique patient-date-file combinations")
        except Exception as e:
            print(f"Warning: Could not read existing output file {output_file}: {e}")
            print("Will proceed with full processing")
    
    # Get all DICOM files
    all_dicom_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if (file.lower().endswith(('.dcm', '.dicom', '.ima')) or 
                '.' not in file or
                _is_dicom_file(file_path)):
                all_dicom_files.append(file_path)
    
    if not all_dicom_files:
        print(f"No DICOM files found in {folder_path}")
        return existing_data if output_format == 'dataframe' and not existing_data.empty else {}
    
    # Filter out already processed files using robust checking
    dicom_files = []
    skipped_count = 0
    
    for file_path in all_dicom_files:
        relative_path = os.path.relpath(file_path, folder_path)
        
        # More robust check using patient ID and acquisition date
        if processed_combinations:
            try:
                # Extract basic metadata without full processing
                patient_id, acquisition_date = _extract_basic_metadata(file_path)
                combo = (str(patient_id), str(acquisition_date), relative_path)
                
                if combo in processed_combinations:
                    skipped_count += 1
                    continue
                    
            except Exception as e:
                print(f"Warning: Could not extract metadata from {file_path}: {e}")
                # If we can't extract metadata, include the file for processing
        
        dicom_files.append(file_path)
    
    if not dicom_files:
        print(f"All {len(all_dicom_files)} DICOM files have already been processed")
        return existing_data if output_format == 'dataframe' and not existing_data.empty else {}
    
    print(f"Found {len(all_dicom_files)} total DICOM files, {len(dicom_files)} new files to process ({skipped_count} already processed)")
    
    # Extract perfusion parameters from each new file
    extracted_data = {}
    
    for i, dicom_file in enumerate(dicom_files):
        try:
            print(f"Processing file {i+1}/{len(dicom_files)}: {os.path.basename(dicom_file)}")
            
            perfusion_data = extract_single_file_perfusion_params(dicom_file)
            relative_path = os.path.relpath(dicom_file, folder_path)
            extracted_data[relative_path] = perfusion_data
            
        except Exception as e:
            print(f"Error processing {dicom_file}: {str(e)}")
            continue
    
    # Format output
    if output_format == 'dataframe':
        new_result = _convert_to_dataframe(extracted_data)
        # Drop rows with empty or NaN volume values
        new_result = new_result.dropna(subset=['volume'])
        new_result = new_result[new_result['volume'] != '']  # Also remove empty strings
        new_result = new_result.reset_index(drop=True)
        
        # Combine with existing data if available
        if not existing_data.empty:
            # Ensure data type consistency before combining
            existing_data = _normalize_dataframe_types(existing_data)
            new_result = _normalize_dataframe_types(new_result)
            
            result = pd.concat([existing_data, new_result], ignore_index=True)
            print(f"Combined {len(existing_data)} existing records with {len(new_result)} new records")
        else:
            result = new_result

        # Drop duplicate rows to remove OCR duplicates (after type normalization)
        result = result.drop_duplicates(subset=['patient_id', 'acquisition_date', 'parameter_type', 'threshold', 'volume'], keep='first')
            
    elif output_format == 'json':
        result = json.dumps(extracted_data, indent=2, default=str)
    else:
        result = extracted_data
    
    # Save to file if requested
    if save_to_file:
        if output_file is None:
            output_file = f"perfusion_parameters.{output_format if output_format != 'dict' else 'json'}"
        
        _save_results(result, output_file, output_format)
        print(f"Results saved to: {output_file}")
    
    return result


def extract_single_file_perfusion_params(file_path: str) -> Dict[str, Any]:
    """
    Extract perfusion parameters from a single DICOM file.
    
    Returns:
    --------
    Dict containing:
    - file_info: basic file information
    - patient_id: DICOM PatientID field
    - acquisition_date: DICOM acquisition/study/series date
    - cbf_parameters: list of CBF measurements with thresholds and volumes
    - tmax_parameters: list of Tmax measurements with thresholds and volumes
    - cbv_parameters: list of CBV measurements with thresholds and volumes
    - raw_text: full OCR text for debugging
    """
    
    # Read DICOM file
    ds = pydicom.dcmread(file_path, force=True)
    
    # Extract PatientID
    patient_id = None
    try:
        patient_id = str(ds.PatientID) if hasattr(ds, 'PatientID') and ds.PatientID else None
    except Exception as e:
        print(f"Warning: Could not extract PatientID from {file_path}: {e}")
    
    # Extract acquisition date
    acquisition_date = None
    try:
        if hasattr(ds, 'AcquisitionDate') and ds.AcquisitionDate:
            acquisition_date = str(ds.AcquisitionDate)
        elif hasattr(ds, 'StudyDate') and ds.StudyDate:
            acquisition_date = str(ds.StudyDate)
        elif hasattr(ds, 'SeriesDate') and ds.SeriesDate:
            acquisition_date = str(ds.SeriesDate)
    except Exception as e:
        print(f"Warning: Could not extract acquisition date from {file_path}: {e}")
    
    result = {
        'file_path': file_path,
        'patient_id': patient_id,
        'acquisition_date': acquisition_date,
        'cbf_parameters': [],
        'tmax_parameters': [],
        'cbv_parameters': [],
        'raw_text': '',
        'error': None
    }
    
    # Check if file has pixel data
    if not hasattr(ds, 'pixel_array') or ds.pixel_array is None:
        print(f"No pixel data found in {file_path}")
        return result
    
    try:
        # Get pixel array and prepare for OCR
        pixel_array = ds.pixel_array
        
        # Prepare image for OCR with aggressive preprocessing for text extraction
        image = _prepare_image_for_perfusion_ocr(pixel_array)
        
        # Perform OCR with high confidence settings
        raw_text = pytesseract.image_to_string(
            image,
            config='--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:%><=()[]ml '
        )
        
        result['raw_text'] = raw_text
        
        # Extract CBF, Tmax, and CBV parameters
        result['cbf_parameters'] = _extract_cbf_parameters(raw_text)
        result['tmax_parameters'] = _extract_tmax_parameters(raw_text)
        result['cbv_parameters'] = _extract_cbv_parameters(raw_text)
        
    except Exception as e:
        print(f"Error extracting from {file_path}: {str(e)}")
        result['error'] = str(e)
    
    return result


def _prepare_image_for_perfusion_ocr(pixel_array: np.ndarray) -> Image.Image:
    """
    Prepare DICOM image specifically for perfusion parameter OCR.
    """
    
    # Normalize to 8-bit
    if pixel_array.dtype != np.uint8:
        if pixel_array.max() > 255:
            pixel_array = ((pixel_array - pixel_array.min()) / 
                          (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        else:
            pixel_array = pixel_array.astype(np.uint8)
    
    # Handle multi-dimensional images
    if len(pixel_array.shape) == 3:
        # Convert to PIL Image
        image = Image.fromarray(pixel_array)
    else:
        # Single frame grayscale
        image = Image.fromarray(pixel_array)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Aggressive preprocessing for text extraction
    # 1. Resize for better OCR
    new_size = (int(image.width * 2.0), int(image.height * 2.0))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # 2. Enhance contrast significantly
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(3.0)  # High contrast
    
    # 3. Enhance sharpness
    sharpness_enhancer = ImageEnhance.Sharpness(image)
    image = sharpness_enhancer.enhance(2.0)
    
    # 4. Convert to grayscale for better thresholding
    gray_image = image.convert('L')
    
    # 5. Apply multiple thresholding techniques
    cv_image = cv2.cvtColor(np.array(gray_image), cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # Apply OTSU thresholding for better text separation
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to PIL
    return Image.fromarray(thresh).convert('RGB')


def _extract_cbf_parameters(text: str) -> List[Dict[str, Any]]:
    """
    Extract CBF (Cerebral Blood Flow) parameters with thresholds and volumes.
    
    Looking for patterns like:
    - "CBF30% volume: 21 ml"
    - "CBF34% volume: 6 ml" 
    - "CBF>30% volume: 4 ml"
    """
    cbf_params = []
    
    # Pattern for CBF with threshold and volume
    # Matches: CBF30%, CBF34%, CBF>30%, CBF<34%, etc.
    cbf_patterns = [
        r'CBF\s*([><]?\s*\d+\.?\d*)\s*%\s*volume\s*:?\s*(\d+\.?\d*)\s*ml',
        r'CBF\s*([><]?\s*\d+\.?\d*)\s*%\s*volumes?\s*:?\s*(\d+\.?\d*)\s*ml',
        r'CBF([><]?\d+\.?\d*)%\s*volume\s*:?\s*(\d+\.?\d*)\s*ml',
        r'CBF([><]?\d+\.?\d*)%\s*volumes?\s*:?\s*(\d+\.?\d*)\s*ml'
    ]
    
    for pattern in cbf_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            threshold = match.group(1).strip()
            volume = match.group(2).strip()
            
            # Clean up threshold (remove spaces)
            threshold = threshold.replace(' ', '')
            
            cbf_param = {
                'parameter': 'CBF',
                'threshold': threshold,
                'volume': float(volume),
                'unit': 'ml',
                'raw_match': match.group(0)
            }
            cbf_params.append(cbf_param)
    
    return cbf_params


def _extract_tmax_parameters(text: str) -> List[Dict[str, Any]]:
    """
    Extract Tmax (Time to Maximum) parameters with thresholds and volumes.
    
    Looking for patterns like:
    - "Tmax>6.0s volume: 21 ml"
    - "Tmax>4.0s volume: 6 ml"
    - "TIMAX10.0s volume: 4 ml"
    """
    tmax_params = []
    
    # Pattern for Tmax with threshold and volume
    tmax_patterns = [
        r'Tmax\s*([><]?\s*\d+\.?\d*)\s*s\s*volume\s*:?\s*(\d+\.?\d*)\s*ml',
        r'Tmax\s*([><]?\s*\d+\.?\d*)\s*s\s*volumes?\s*:?\s*(\d+\.?\d*)\s*ml',
        r'TIMAX\s*([><]?\s*\d+\.?\d*)\s*s\s*volume\s*:?\s*(\d+\.?\d*)\s*ml',
        r'TIMAX([><]?\d+\.?\d*)s\s*volume\s*:?\s*(\d+\.?\d*)\s*ml',
        r'Tmax([><]?\d+\.?\d*)s\s*volume\s*:?\s*(\d+\.?\d*)\s*ml'
    ]
    
    for pattern in tmax_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            threshold = match.group(1).strip()
            volume = match.group(2).strip()
            
            # Clean up threshold
            threshold = threshold.replace(' ', '')
            
            tmax_param = {
                'parameter': 'Tmax',
                'threshold': threshold,
                'volume': float(volume),
                'unit': 'ml',
                'raw_match': match.group(0)
            }
            tmax_params.append(tmax_param)
    
    return tmax_params


def _extract_cbv_parameters(text: str) -> List[Dict[str, Any]]:
    """
    Extract CBV (Cerebral Blood Volume) parameters with thresholds and volumes.
    
    Looking for patterns like:
    - "CBV<34% volume: 21 ml"
    - "CBV<20% volume: 6 ml" 
    - "CBV>30% volume: 4 ml"
    - "CBV<2.0ml/100g volume: 21 ml" (alternative unit format)
    """
    cbv_params = []
    
    # Pattern for CBV with threshold and volume
    # Prioritize percentage patterns as they're more common
    cbv_patterns = [
        # Percentage patterns (most common)
        r'CBV\s*([><]?\s*\d+\.?\d*)\s*%\s*volume\s*:?\s*(\d+\.?\d*)\s*ml',
        r'CBV\s*([><]?\s*\d+\.?\d*)\s*%\s*volumes?\s*:?\s*(\d+\.?\d*)\s*ml',
        r'CBV([><]?\d+\.?\d*)%\s*volume\s*:?\s*(\d+\.?\d*)\s*ml',
        r'CBV([><]?\d+\.?\d*)%\s*volumes?\s*:?\s*(\d+\.?\d*)\s*ml',
        # ml/100g unit patterns (alternative format)
        r'CBV\s*([><]?\s*\d+\.?\d*)\s*ml/100g\s*volume\s*:?\s*(\d+\.?\d*)\s*ml',
        r'CBV\s*([><]?\s*\d+\.?\d*)\s*ml/100g\s*volumes?\s*:?\s*(\d+\.?\d*)\s*ml',
        r'CBV([><]?\d+\.?\d*)ml/100g\s*volume\s*:?\s*(\d+\.?\d*)\s*ml',
        r'CBV([><]?\d+\.?\d*)ml/100g\s*volumes?\s*:?\s*(\d+\.?\d*)\s*ml'
    ]
    
    for pattern in cbv_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            threshold = match.group(1).strip()
            volume = match.group(2).strip()
            
            # Clean up threshold (remove spaces)
            threshold = threshold.replace(' ', '')
            
            cbv_param = {
                'parameter': 'CBV',
                'threshold': threshold,
                'volume': float(volume),
                'unit': 'ml',
                'raw_match': match.group(0)
            }
            cbv_params.append(cbv_param)
    
    return cbv_params


def _extract_basic_metadata(file_path: str) -> tuple:
    """
    Extract basic metadata (patient_id, acquisition_date) from DICOM file without full processing.
    
    Returns:
    --------
    tuple: (patient_id, acquisition_date)
    """
    try:
        ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
        
        # Extract PatientID
        patient_id = None
        try:
            patient_id = str(ds.PatientID) if hasattr(ds, 'PatientID') and ds.PatientID else None
        except Exception:
            pass
        
        # Extract acquisition date
        acquisition_date = None
        try:
            if hasattr(ds, 'AcquisitionDate') and ds.AcquisitionDate:
                acquisition_date = str(ds.AcquisitionDate)
            elif hasattr(ds, 'StudyDate') and ds.StudyDate:
                acquisition_date = str(ds.StudyDate)
            elif hasattr(ds, 'SeriesDate') and ds.SeriesDate:
                acquisition_date = str(ds.SeriesDate)
        except Exception:
            pass
            
        return patient_id, acquisition_date
        
    except Exception:
        return None, None


def _is_dicom_file(file_path: str) -> bool:
    """Check if a file is a DICOM file."""
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False


def Pourquoi ?_normalize_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame column types to ensure consistency for duplicate detection.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to normalize
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with normalized column types
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Normalize patient_id to string type
    if 'patient_id' in df.columns:
        df['patient_id'] = df['patient_id'].astype(str)
    
    # Normalize acquisition_date to string type
    if 'acquisition_date' in df.columns:
        df['acquisition_date'] = df['acquisition_date'].astype(str)
    
    # Normalize file_path to string type
    if 'file_path' in df.columns:
        df['file_path'] = df['file_path'].astype(str)
    
    # Normalize parameter_type to string type
    if 'parameter_type' in df.columns:
        df['parameter_type'] = df['parameter_type'].astype(str)
    
    # Normalize threshold to string type (since it can contain '<', '>', etc.)
    if 'threshold' in df.columns:
        df['threshold'] = df['threshold'].astype(str)
    
    # Normalize volume to float type
    if 'volume' in df.columns:
        # Handle non-numeric values by converting to NaN first
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    
    # Normalize unit to string type
    if 'unit' in df.columns:
        df['unit'] = df['unit'].astype(str)
    
    return df


def _convert_to_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """Convert extracted perfusion data to a pandas DataFrame."""
    rows = []
    
    for file_path, file_data in data.items():
        base_row = {
            'file_path': file_path,
            'patient_id': file_data.get('patient_id'),
            'acquisition_date': file_data.get('acquisition_date'),
            'error': file_data.get('error')
        }
        
        # Add CBF, Tmax, and CBV parameters
        cbf_params = file_data.get('cbf_parameters', [])
        tmax_params = file_data.get('tmax_parameters', [])
        cbv_params = file_data.get('cbv_parameters', [])
        
        if cbf_params or tmax_params or cbv_params:
            # Create rows for each parameter
            for cbf in cbf_params:
                row = base_row.copy()
                row.update({
                    'parameter_type': 'CBF',
                    'threshold': cbf['threshold'],
                    'volume': cbf['volume'],
                    'unit': cbf['unit'],
                    'raw_match': cbf['raw_match']
                })
                rows.append(row)
            
            for tmax in tmax_params:
                row = base_row.copy()
                row.update({
                    'parameter_type': 'Tmax',
                    'threshold': tmax['threshold'],
                    'volume': tmax['volume'],
                    'unit': tmax['unit'],
                    'raw_match': tmax['raw_match']
                })
                rows.append(row)
            
            for cbv in cbv_params:
                row = base_row.copy()
                row.update({
                    'parameter_type': 'CBV',
                    'threshold': cbv['threshold'],
                    'volume': cbv['volume'],
                    'unit': cbv['unit'],
                    'raw_match': cbv['raw_match']
                })
                rows.append(row)
        else:
            # No parameters found, add empty row
            row = base_row.copy()
            row.update({
                'parameter_type': None,
                'threshold': None,
                'volume': None,
                'unit': None,
                'raw_match': None
            })
            rows.append(row)
    
    return pd.DataFrame(rows)


def _save_results(results, output_file: str, output_format: str):
    """Save results to file. For dataframe format, this overwrites the entire file with combined data."""
    if output_format == 'dataframe':
        if output_file.endswith('.csv'):
            results.to_csv(output_file, index=False)
        elif output_file.endswith('.xlsx'):
            results.to_excel(output_file, index=False)
        else:
            results.to_csv(f"{output_file}.csv", index=False)
    
    elif output_format == 'json':
        with open(output_file, 'w') as f:
            if isinstance(results, str):
                f.write(results)
            else:
                json.dump(results, f, indent=2, default=str)
    
    else:  # dict format
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command line interface parser."""
    parser = argparse.ArgumentParser(
        description='Extract perfusion parameters (CBF, Tmax, CBV) from DICOM images using OCR. Supports incremental processing to avoid reprocessing existing files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from folder and save to CSV (default behavior)
  python extract_perfusion_parameters.py /path/to/dicom/folder
  
  # Extract and display only (no file saving)
  python extract_perfusion_parameters.py /path/to/dicom/folder --no-save
  
  # Save to specific file path
  python extract_perfusion_parameters.py /path/to/dicom/folder --output-file /path/to/results.csv
  
  # Incremental processing: if results.csv exists, only process new files
  python extract_perfusion_parameters.py /path/to/dicom/folder --output-file results.csv
  python extract_perfusion_parameters.py /path/to/new/dicom/folder --output-file results.csv  # appends new results
  
  # Verbose output with custom filename
  python extract_perfusion_parameters.py /path/to/dicom/folder --output-file my_results.csv --verbose

Incremental Processing:
  If an output file already exists, the script will:
  - Read the existing file to identify already processed files
  - Only process DICOM files not already in the output
  - Append new results to the existing data
  - Save the combined dataset to the same file
        """
    )
    
    parser.add_argument(
        'folder_path',
        help='Path to folder containing DICOM files'
    )
    
    parser.add_argument(
        '--no-save', 
        action='store_true',
        help='Do not save results to file (display only)'
    )
    
    parser.add_argument(
        '--output-file', '-o',
        help='Output file path (default: perfusion_parameters.csv in input directory). If file exists, only new files will be processed and results appended.'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    return parser


def main():
    """Main CLI function."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Validate folder path
    if not os.path.exists(args.folder_path):
        print(f"❌ Error: Folder path does not exist: {args.folder_path}")
        sys.exit(1)
    
    if not os.path.isdir(args.folder_path):
        print(f"❌ Error: Path is not a directory: {args.folder_path}")
        sys.exit(1)
    
    try:
        if not args.quiet:
            print("🔍 Starting perfusion parameter extraction...")
            print(f"📁 Processing folder: {args.folder_path}")
        
        # Determine if we should save to file (default is True unless --no-save is specified)
        save_to_file = not args.no_save
        
        # Generate default output file path in input directory if not specified
        output_file = args.output_file
        if save_to_file and output_file is None:
            output_file = os.path.join(args.folder_path, "perfusion_parameters.csv")
        
        # Extract perfusion parameters (always use dataframe format)
        results = extract_perfusion_parameters(
            folder_path=args.folder_path,
            output_format='dataframe',
            save_to_file=save_to_file,
            output_file=output_file
        )
        
        # Display results
        if not args.quiet and isinstance(results, pd.DataFrame):
            print("\n📊 Extraction Summary:")
            print(f"   Total measurements: {len(results)}")
            print(f"   Unique patients: {results['patient_id'].nunique()}")
            print(f"   Files processed: {results['file_path'].nunique()}")
            print(f"   CBF measurements: {len(results[results['parameter_type'] == 'CBF'])}")
            print(f"   Tmax measurements: {len(results[results['parameter_type'] == 'Tmax'])}")
            print(f"   CBV measurements: {len(results[results['parameter_type'] == 'CBV'])}")
            
            if args.verbose:
                print("\n📋 First 10 measurements:")
                print(results[['patient_id', 'acquisition_date', 'file_path', 'parameter_type', 'threshold', 'volume']].head(10))
                       
        if not args.quiet:
            print("\n✅ Extraction completed successfully!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
