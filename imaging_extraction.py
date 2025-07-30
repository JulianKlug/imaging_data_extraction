import pandas as pd
import pyautogui
import argparse

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 1

JUSTIFICATION = 'CCER2016-01445'

COORDINATES = {
    'window': (1000, 236),
    'search_bar': (421, 293),  # Example coordinates for the search bar
    'reinitialise': (527, 585),
    'patient_id_field': (566, 430),  # Example coordinates for the patient ID field
    'search_button': (414, 586),  # Example coordinates for the search button
    'justification_field': (918, 683),  # Example coordinates for the justification field
    'validation_button': (1399, 760),  # Example coordinates for the validation button
    'all_imaging_data_button': (376, 462),  # Example coordinates for the all imaging data button
    'export_button': (432, 424),  # Example coordinates for the export button
    'transfer_button': (1243, 693),
    'confirm_transfer_button': (1212, 729),
    'close_patient_button': (653, 291)
}

def extract_patient(patient_id):
    print(f'Extracting data for patient {patient_id}...')
    # 0. focus on window, click on search bar and initialise
    pyautogui.click(COORDINATES['window'])
    pyautogui.click(COORDINATES['search_bar'])
    pyautogui.click(COORDINATES['reinitialise'])

    # 1. click on patient id field
    pyautogui.click(COORDINATES['patient_id_field'])

    # 2. enter patient id
    pyautogui.typewrite(str(patient_id))

    # 3. click on search button
    pyautogui.click(COORDINATES['search_button'])

    # 4. click on justification field
    pyautogui.click(COORDINATES['justification_field'])

    # 5. enter justification
    pyautogui.typewrite(JUSTIFICATION)

    # 6. click on validation
    pyautogui.click(COORDINATES['validation_button'])

    # 7. click on all imaging data
    pyautogui.click(COORDINATES['all_imaging_data_button'])

    # 8. click on export button
    pyautogui.click(COORDINATES['export_button'])

    # 9. click on transfer button
    pyautogui.click(COORDINATES['transfer_button'])

    # 10. click on export button
    pyautogui.click(COORDINATES['confirm_transfer_button'])

    # 11. close patient window
    pyautogui.click(COORDINATES['close_patient_button'])


def extract_n_patients(number_of_patients_to_extract, target_patients_path):
    target_list = pd.read_csv(target_patients_path)

    extracted_patients = []
    for pidx in range(number_of_patients_to_extract):
        if pidx > number_of_patients_to_extract - 1:
            break
        patient_id = target_list['patient_id'].iloc[pidx]
        patient_data = extract_patient(patient_id)
        extracted_patients.append(patient_data)

    return extracted_patients


def main():
    parser = argparse.ArgumentParser(description='Extract patient data.')
    parser.add_argument('-n', '--number_of_patients_to_extract', type=int, required=True,
                        help='Number of patients to extract')
    parser.add_argument('-t', '--target_patients_path', type=str, required=True,
                        help='Path to the CSV file containing target patients')

    args = parser.parse_args()

    # show instructions on how to exit the pyautogui script
    print("To stop the script, move your mouse to the top-left corner of the screen.")

    extracted_patients = extract_n_patients(args.number_of_patients_to_extract, args.target_patients_path)
    print(f'Extracted {len(extracted_patients)} patients.')

if __name__ == '__main__':
    main()
