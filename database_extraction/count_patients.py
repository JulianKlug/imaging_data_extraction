import pandas as pd
import pydicom
import os
import shutil

def count_patients(folder_path):
    patient_ids = set()
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dcm"):
                file_path = os.path.join(root, file)
                # ds_pid = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
                ds_pid = pydicom.dcmread(file_path, force=True)
                try:
                    pid = ds_pid.PatientID
                    patient_ids.add(pid)
                except AttributeError:
                    print(f"File {file_path} does not contain PatientID")
                    print(ds_pid)
    return len(patient_ids)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Count DICOM patients in a folder")
    parser.add_argument("-f", "--folder", type=str, help="Path to the folder containing DICOM files")
    args = parser.parse_args()

    num_patients = count_patients(args.folder)
    print(f"Number of patients: {num_patients}")