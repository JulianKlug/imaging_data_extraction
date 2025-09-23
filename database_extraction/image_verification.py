import pydicom
import os
import shutil

def is_rapid_file(ds):
    """Check if the DICOM file meets RAPID identification criteria.
        # RAPID file identification criteria
    # Manufacturer contains ISCHEMAVIEW
    # StationName: contains RAPID
    # SeriesDescription contains RAPID
    # ManufacturersModelName contains RAPID
    # ImageComments contains RAPID
    
    """
    is_rapid = False
    try:
        if 'ISCHEMAVIEW' in ds.Manufacturer:
            is_rapid = True
    except:
        pass
        
    try:
        if 'RAPID' in ds.StationName:
            is_rapid = True
    except:
        pass
        
    try:
        if 'RAPID' in ds.SeriesDescription:
            is_rapid = True
    except:
        pass
        
    try:
        if 'RAPID' in ds.ManufacturersModelName:
            is_rapid = True
    except:
        pass
        
    try:
        if 'RAPID' in ds.ImageComments:
            is_rapid = True
    except:
        pass

    return is_rapid


def verify_rapid_presence(pid, dicom_db_path, output_dir, delete_unused=False, verbose=False):
    
    # loop through subdirectories in dicom_db_path
    for root, dirs, files in os.walk(dicom_db_path):
        for file in files:
            if file.endswith('.dcm'):
                file_path = os.path.join(root, file)
                try:
                    ds_pid = pydicom.dcmread(file_path, stop_before_pixels=True)
                    if ds_pid.PatientID == pid:
                        ds = pydicom.dcmread(file_path)
                        if verbose:
                            print(f"Found DICOM file for patient {pid} at {file_path}")

                        # check if the DICOM file meets any criteria
                        if is_rapid_file(ds):
                            if verbose:
                                print(f"RAPID file identified for patient {pid}: {file_path}")
                            # Optionally, copy the file to output_dir
                            shutil.copy(file_path, output_dir)
                            return True
                        
                        else:
                            if verbose:
                                print(f"File {file_path} does not meet RAPID criteria for patient {pid}")
                        if delete_unused:
                            os.remove(file_path)

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    if verbose:
        print(f"No RAPID files found for patient {pid} in {dicom_db_path}")
    return False



def find_pids_already_extracted(dicom_db_path, output_dir, delete_unused=False, verbose=False):
    """
    Find all patient IDs that have already been extracted.
    """
    if verbose:
        print(f"Searching for already extracted patient IDs in {dicom_db_path}...")

    pids_with_rapid = []
    pids_screened = set()
    for root, dirs, files in os.walk(dicom_db_path):
        for file in files:
            if file.endswith('.dcm'):
                file_path = os.path.join(root, file)
                try:
                    ds = pydicom.dcmread(file_path)
                    pid = ds.PatientID
                    pids_screened.add(pid)
                    if pids_with_rapid(ds):
                        pids_with_rapid.append(pid)
                        # copy the file to output_dir
                        shutil.copy(file_path, output_dir)
                        if verbose:
                            print(f"RAPID file identified for patient {pid}: {file_path}")
                    if delete_unused:
                        os.remove(file_path)
                except Exception as e:
                    if verbose:
                        print(f"Error reading {file_path}: {e}")
    # Remove duplicates
    pids_with_rapid = list(set(pids_with_rapid))
    pids_without_rapid = pids_screened - set(pids_with_rapid)

    if verbose:
        print(f"Found {len(pids_with_rapid)} unique patient IDs with RAPID files in {dicom_db_path}.")

    return pids_with_rapid, pids_without_rapid
