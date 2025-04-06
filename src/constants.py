from dataclasses import dataclass


@dataclass
class ConstantsDataSetPaths:
    """Constants for the data set paths."""

    path_file_pdf_data_set = "data/CONTRATO_AP000000718.pdf"
    path_file_train_questions = "data/modeling/train_questions.csv"
    path_file_test_questions = "data/modeling/test_questions.csv"
    extracted_text = "data/extracted_text.pkl"


DATA_SET_PATHS = ConstantsDataSetPaths()
