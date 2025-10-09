import os
import sys
import pandas as pd
import pyautogui
import argparse
import time
from image_verification import find_pids_already_extracted, verify_rapid_presence
from list_open_windows_mac import list_open_windows
from launch_pacs import launch_osirix_pacs, kill_osirix_pacs

pyautogui.FAILSAFE = True

SINGLE_MODALITY = True

SLEEP_TIME = 10
WAIT_FOR_DOWNLOAD_TIME = 45
WAIT_BEFORE_SEARCH_TIME = 0.5
WAIT_BEFORE_SELECT_ALL_TIME = 2.25
WAIT_EVERY_N_PATIENTS = 180
WAIT_FOR_INCOMING_DB_TIME = 1.5
DEFAULT_PYAUTOGUI_PAUSE = 0.6
SAFE_MODE_PYAUTOGUI_PAUSE = 0.1

BATCH_SIZE = 1

if SINGLE_MODALITY:
    WAIT_FOR_DOWNLOAD_TIME = 0

JUSTIFICATION = 'CCER2016-01445'


COORDINATES = {
    "reference": (378, 177), # reference of close window cross 
    'window': (1000, 236),
    'search_bar': (421, 293),  # coordinates for the search bar
    'reinitialise': (527, 585),
    'patient_id_field': (566, 430),  # coordinates for the patient ID field
    'modality_field': (1274, 531),  # coordinates for the modality field
    'search_button': (414, 586),  # coordinates for the search button
    'justification_field': (918, 683),  # coordinates for the justification field
    'validation_button': (1399, 760),  # coordinates for the validation button
    'all_imaging_data_button': (376, 462),  # coordinates for the all imaging data button
    'single_modality_all_imaging_data_button': (371, 675),  # coordinates for the all imaging data button in single modality
    'export_button': (432, 424),  # coordinates for the export button
    "single_modality_export_button": (428, 639),  # coordinates for the export button in single modality
    'transfer_button': (1243, 693),
    'confirm_transfer_button': (1212, 729),
    'confirm_transfer_button_with_error': (1208, 773),
    'close_patient_button': (653, 291),
    'osirix_popup_ok': (1118,525)
}

def extract_patient(patient_id, 
                    dicom_db_path,
                    output_dir,
                    path_to_identifiers_file,
                    delete_unused=False,
                    safe_mode=True,
                    verbose=False):
    
    # verify that osirix is open
    ensure_osirix_open(path_to_identifiers_file, verbose=verbose)

    verify_modal_failsafe(safe_mode=safe_mode)
    print(f'Extracting data for patient {patient_id}...')
    # 0. focus on window, click on search bar and initialise
    if verbose:
        print(f'Focusing on the window and initialising search...')
    pyautogui.click(COORDINATES['window'])
    verify_modal_failsafe(safe_mode=safe_mode)

    pyautogui.click(COORDINATES['search_bar'])
    verify_modal_failsafe(safe_mode=safe_mode)
    pyautogui.click(COORDINATES['reinitialise'])

    # 1. click on patient id field
    if verbose:
        print(f'Clicking on patient ID field')
    verify_modal_failsafe(safe_mode=safe_mode)
    pyautogui.click(COORDINATES['patient_id_field'])

    # 2. enter patient id
    if verbose:
        print(f'Entering patient ID')
    verify_modal_failsafe(safe_mode=safe_mode)
    pyautogui.typewrite(str(patient_id))

    if SINGLE_MODALITY:
        # 3. enter modality
        if verbose:
            print('Specifying modality')
        verify_modal_failsafe(safe_mode=safe_mode)
        pyautogui.click(COORDINATES['modality_field'])
        pyautogui.typewrite('CT')
        pyautogui.press('enter')

    # 4. click on search button
    time.sleep(WAIT_BEFORE_SEARCH_TIME)
    if verbose:
        print(f'Clicking on search button')
    verify_modal_failsafe(safe_mode=safe_mode)
    pyautogui.click(COORDINATES['search_button'])

    if SINGLE_MODALITY:
        # click on all imaging data button
        time.sleep(WAIT_BEFORE_SELECT_ALL_TIME)
        if verbose:
            print(f'Clicking on all imaging data button')
        verify_modal_failsafe(safe_mode=safe_mode)
        pyautogui.click(COORDINATES['single_modality_all_imaging_data_button'])
        pyautogui.click(COORDINATES['single_modality_export_button'])

    else:
        # 5. click on justification field
        if verbose:
            print(f'Clicking on justification field')
        verify_modal_failsafe(safe_mode=safe_mode)
        pyautogui.click(COORDINATES['justification_field'])

        # 6. enter justification
        if verbose:
            print(f'Entering justification')
        verify_modal_failsafe(safe_mode=safe_mode)
        pyautogui.typewrite(JUSTIFICATION)

        # 7. click on validation
        if verbose:
            print(f'Clicking on validation button')
        verify_modal_failsafe(safe_mode=safe_mode)
        pyautogui.click(COORDINATES['validation_button'])

        # 8. click on all imaging data
        if verbose:
            print(f'Clicking on all imaging data button')
        verify_modal_failsafe(safe_mode=safe_mode)
        pyautogui.click(COORDINATES['all_imaging_data_button'])

        # 9. click on export button
        if verbose:
            print(f'Clicking on export button')
        verify_modal_failsafe(safe_mode=safe_mode)
        pyautogui.click(COORDINATES['export_button'])

    # 10. click on transfer button
    if verbose:
        print(f'Clicking on transfer button')
    verify_modal_failsafe(safe_mode=safe_mode)
    pyautogui.click(COORDINATES['transfer_button'])

    # 11. click on confirm transfer button
    if verbose:
        print(f'Clicking on confirm transfer button')
    verify_modal_failsafe(safe_mode=safe_mode)
    pyautogui.click(COORDINATES['confirm_transfer_button'])

    verify_modal_failsafe(safe_mode=safe_mode)
    pyautogui.click(COORDINATES['confirm_transfer_button_with_error'])

    if not SINGLE_MODALITY:
        # wait for transfer dialog to disappear (sleep)
        time.sleep(SLEEP_TIME)
    
        # 12. close patient window
        if verbose:
            print(f'Clicking on close patient button')
        verify_modal_failsafe(safe_mode=safe_mode)
        pyautogui.click(COORDINATES['close_patient_button'])

    time.sleep(WAIT_FOR_DOWNLOAD_TIME)
  
    return 

def ensure_osirix_open(path_to_identifiers_file,
                      verbose=False):
    windows = list_open_windows()
    osirix_windows = [w for w in windows if w[0] == "OsiriX MD"]    
    if len(osirix_windows) < 2:
        if path_to_identifiers_file is None:
            raise Exception('Osirix or Compacs are not open!')
        else:
            print('Osirix or Compacs are not open! Attempting to re-open Osirix...')
            kill_osirix_pacs()
            launch_osirix_pacs(path_to_identifiers_file, verbose=verbose)
    return 

def verify_modal_failsafe(safe_mode=True):
    if not safe_mode:
        return
    if verify_modal_presence():
        sys.exit("Error message")


def verify_modal_presence():
    """
    Verify if a modal dialog is present on the screen.
    """
    # modals to verify 
    # check all png files in modal directory (current file is in the same directory as the modal directory)
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    modal_directory = os.path.join(current_file_directory, 'modals')
    modal_files = [f for f in os.listdir(modal_directory) if f.endswith('.png')]
    for modal_file in modal_files:
        modal_path = os.path.join(modal_directory, modal_file)
        try:
            if pyautogui.locateOnScreen(modal_path) is not None:
                print(f'Modal dialog {modal_file} is present on the screen.')
                return True
        except Exception as e:
            pass  # If the image is not found, continue to the next one

    return False

def extract_target_and_clean_dicom_db(dicom_db_path, output_dir, 
                            already_extracted_list_path, target_list, 
                            already_extracted_df,
                            delete_unused=False, verbose=False):
    """
    Extract RAPID files for patients already in the dicom_db_path and clean up the DICOM DB.
    """

    # find all pids already extracted in the dicom_db_path and copy their files to output_dir
    found_pids, pids_without_rapid = find_pids_already_extracted(dicom_db_path, output_dir, delete_unused, verbose)
    target_list = target_list[~target_list['patient_id'].isin(found_pids)]
    # add found pids to already_extracted_df
    for pid in found_pids:
        # if pid already in already_extracted_df, drop previous entry
        already_extracted_df = already_extracted_df[already_extracted_df['patient_id'] != pid]
        already_extracted_df = pd.concat([already_extracted_df, pd.DataFrame({'patient_id': [pid], 
                                                                              'RAPID_found': [True]})], ignore_index=True)
    for pid in pids_without_rapid:
        if pid in already_extracted_df['patient_id'].values:
            continue
        already_extracted_df = pd.concat([already_extracted_df, pd.DataFrame({'patient_id': [pid], 
                                                                              'RAPID_found': [0]})], ignore_index=True)

    already_extracted_df = already_extracted_df.drop_duplicates(subset=['patient_id'])
    already_extracted_df.to_csv(already_extracted_list_path, index=False)

    if delete_unused:
        # delete all subfolders in dicom_db_path 
        for subfolder in os.listdir(dicom_db_path):
            subfolder_path = os.path.join(dicom_db_path, subfolder)
            if os.path.isdir(subfolder_path):
                try:
                    os.rmdir(subfolder_path)
                except OSError as e:
                    if verbose:
                        print(f"Error deleting folder {subfolder_path}: {e}")

        # delete Database.sql file in the parent directory of dicom_db_path
        os.remove(os.path.join(os.path.dirname(dicom_db_path),'Database.sql'))

    if verbose:
        # print number of patients already extracted, and number of patients to extract
        print(f'Number of patients already extracted: {len(already_extracted_df)} (remaining {len(target_list)})')

    return target_list, already_extracted_df


def extract_n_patients(number_of_patients_to_extract, target_patients_path,
                       dicom_db_path, output_dir, delete_unused, 
                       path_to_identifiers_file,
                       already_extracted_list_path=None,
                       incoming_db_path=None, 
                       safe_mode=True, verbose=False):
    target_list = pd.read_csv(target_patients_path)
    
    if already_extracted_list_path is not None:
        already_extracted_df = pd.read_csv(already_extracted_list_path)
        target_list = target_list[~target_list['patient_id'].isin(already_extracted_df['patient_id'])]
    else:
        already_extracted_list_path = os.path.join(os.path.dirname(target_patients_path), 'already_extracted.csv')
        if os.path.isfile(already_extracted_list_path):
            already_extracted_df = pd.read_csv(already_extracted_list_path)
            target_list = target_list[~target_list['patient_id'].isin(already_extracted_df['patient_id'])]
        else:
            already_extracted_df = pd.DataFrame()
    
    # extract rapid files for patients already in dicom_db_path
    target_list, already_extracted_df = extract_target_and_clean_dicom_db(dicom_db_path, output_dir,
                                                                        already_extracted_list_path, target_list,
                                                                        already_extracted_df,
                                                                        delete_unused=delete_unused, verbose=verbose)

    iteration_counter = 0
    for pidx in range(number_of_patients_to_extract):
        if pidx > number_of_patients_to_extract - 1:
            break

        # random sample patient_id
        patient_id = target_list['patient_id'].sample(n=1).values[0]

        extract_patient(patient_id, 
                        dicom_db_path=dicom_db_path,
                        output_dir=output_dir,
                        path_to_identifiers_file=path_to_identifiers_file,
                        delete_unused=delete_unused,
                        safe_mode=safe_mode,
                        verbose=verbose)
        iteration_counter += 1
        if iteration_counter == BATCH_SIZE:
            iteration_counter = 0
            if incoming_db_path is not None:
                # wait until incoming_db_path is empty
                print('Waiting for incoming_db_path to be empty...')
                while os.listdir(incoming_db_path):
                    windows = list_open_windows()
                    osirix_windows = [w for w in windows if w[0] == "OsiriX MD"]
                    if len(osirix_windows) > 2:
                        pyautogui.click(COORDINATES['osirix_popup_ok'])
                    ensure_osirix_open(path_to_identifiers_file, verbose=verbose)
                    time.sleep(WAIT_FOR_INCOMING_DB_TIME)

                # extract rapid files for patients already in dicom_db_path
                _, _ = extract_target_and_clean_dicom_db(dicom_db_path, output_dir,
                                                        already_extracted_list_path, target_list,
                                                        already_extracted_df,
                                                        delete_unused=delete_unused, verbose=verbose)
                
            else:
                print(f'Waiting {WAIT_EVERY_N_PATIENTS}s every {BATCH_SIZE} patients...')
                time.sleep(WAIT_EVERY_N_PATIENTS)
            
    return already_extracted_df 


def main():
    parser = argparse.ArgumentParser(description='Extract patient data.')
    parser.add_argument('-n', '--number_of_patients_to_extract', type=int, required=True,
                        help='Number of patients to extract')
    parser.add_argument('-t', '--target_patients_path', type=str, required=True,
                        help='Path to the CSV file containing target patients')
    parser.add_argument('-d', '--dicom_db_path', type=str, required=True,
                        help='Path to the DICOM database')
    parser.add_argument('-p', '--path_to_identifiers_file', type=str, required=False, default=None,
                        help='Path to the file containing PACS identifiers (username and password)')
    parser.add_argument('-idb', '--incoming_db_path', type=str, required=True,
                        help='Path to the incoming DICOM database')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Path to the output directory')
    parser.add_argument('-r', '--delete_unused', action='store_true',
                        help='Delete unused DICOM files')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('-s', '--safe_mode', action='store_true',
                        help='Enable safe mode')
    args = parser.parse_args()

    # show instructions on how to exit the pyautogui script
    print("To stop the script, move your mouse to the top-left corner of the screen.")

    # show modal dialog with instructions
    pyautogui.alert(text='To stop the script, move your mouse to the top-left corner of the screen.',
                    title='Instructions', button='OK')

    if args.safe_mode:
        pyautogui.PAUSE = SAFE_MODE_PYAUTOGUI_PAUSE
    else:
        pyautogui.PAUSE = DEFAULT_PYAUTOGUI_PAUSE

    already_extracted_list_path = None
    if os.path.isfile(os.path.join(os.path.dirname(args.target_patients_path), 'already_extracted.csv')):
        already_extracted_list_path = os.path.join(os.path.dirname(args.target_patients_path), 'already_extracted.csv')

    extracted_patients = extract_n_patients(args.number_of_patients_to_extract, args.target_patients_path,
                                            dicom_db_path=args.dicom_db_path,
                                            output_dir=args.output_dir,
                                            path_to_identifiers_file=args.path_to_identifiers_file,
                                            already_extracted_list_path=already_extracted_list_path,
                                            delete_unused=args.delete_unused,
                                            safe_mode=args.safe_mode,
                                            verbose=args.verbose,
                                            incoming_db_path=args.incoming_db_path)
    print(f'Extracted {len(extracted_patients)} patients.')

if __name__ == '__main__':
    main()
