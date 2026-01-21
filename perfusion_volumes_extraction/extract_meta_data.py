import pydicom


def _extract_basic_metadata(file_path: str) -> tuple:
    """
    Extract basic metadata (patient_id, acquisition_date, acquisition_time) from DICOM file without full processing.

    Returns:
    --------
    tuple: (patient_id, acquisition_date, acquisition_time)
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

        # Extract acquisition time
        acquisition_time = None
        try:
            if hasattr(ds, 'AcquisitionTime') and ds.AcquisitionTime:
                acquisition_time = str(ds.AcquisitionTime)
            elif hasattr(ds, 'StudyTime') and ds.StudyTime:
                acquisition_time = str(ds.StudyTime)
            elif hasattr(ds, 'SeriesTime') and ds.SeriesTime:
                acquisition_time = str(ds.SeriesTime)
        except Exception:
            pass
            
        return patient_id, acquisition_date, acquisition_time
        
    except Exception:
        return None, None, None
    

if __name__ == "__main__":
    # parse command line arguments 
    import argparse
    import os
    import pandas as pd
    parser = argparse.ArgumentParser(description="Extract basic metadata from a DICOM directory.")
    parser.add_argument("directory_path", type=str, help="Path to the DICOM directory.")

    args = parser.parse_args()

    meta_data_df = pd.DataFrame(columns=["patient_id", "file_path", "acquisition_date", "acquisition_time"])

    # run through all files in the directory and extract metadata
    for root, _, files in os.walk(args.directory_path):
        for file in files:
            if file.lower().endswith('.dcm'):
                file_path = os.path.join(root, file)
                patient_id, acquisition_date, acquisition_time = _extract_basic_metadata(file_path)
                temp_df = pd.DataFrame({
                    "patient_id": [patient_id],
                    "file_path": [file],
                    "acquisition_date": [acquisition_date],
                    "acquisition_time": [acquisition_time]
                })
                meta_data_df = pd.concat([meta_data_df, temp_df], ignore_index=True)

    # Save to CSV with timestamp
    output_csv = os.path.join(args.directory_path, f'dicom_metadata_extraction_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv')
    meta_data_df.to_csv(output_csv, index=False)
            