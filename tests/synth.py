"""A small synthetic frame shaped like the UCI extract, for fast tests. Values are invented."""

import numpy as np
import pandas as pd


def frame(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    inpatient = rng.poisson(0.6, n)
    df = pd.DataFrame(
        {
            "encounter_id": np.arange(n) + 1000,
            "patient_nbr": rng.integers(0, n // 2, n),
            "race": rng.choice(["Caucasian", "AfricanAmerican", "Hispanic", None], n),
            "gender": rng.choice(["Female", "Male"], n),
            "age": rng.choice([f"[{a}-{a + 10})" for a in range(30, 100, 10)], n),
            "weight": None,
            "admission_type_id": rng.choice([1, 2, 3, 6], n),
            "discharge_disposition_id": rng.choice([1, 3, 6, 11, 13], n, p=[0.6, 0.2, 0.15, 0.03, 0.02]),
            "admission_source_id": rng.choice([1, 7], n),
            "time_in_hospital": rng.integers(1, 14, n),
            "payer_code": rng.choice(["MC", "HM", None], n),
            "medical_specialty": rng.choice(["InternalMedicine", "Cardiology", None], n),
            "num_lab_procedures": rng.integers(1, 90, n),
            "num_procedures": rng.integers(0, 6, n),
            "num_medications": rng.integers(1, 40, n),
            "number_outpatient": rng.poisson(0.3, n),
            "number_emergency": rng.poisson(0.2, n),
            "number_inpatient": inpatient,
            "diag_1": rng.choice(["428", "250.83", "414", "786", "V57", "996"], n),
            "diag_2": rng.choice(["250", "427", "276", None], n),
            "diag_3": rng.choice(["401", "250.6", "585"], n),
            "number_diagnoses": rng.integers(3, 10, n),
            "max_glu_serum": rng.choice([None, ">300", "Norm"], n),
            "A1Cresult": rng.choice([None, ">8", "Norm"], n),
            "insulin": rng.choice(["No", "Steady", "Up"], n),
            "metformin": rng.choice(["No", "Steady"], n),
            "change": rng.choice(["No", "Ch"], n),
            "diabetesMed": rng.choice(["Yes", "No"], n),
        }
    )
    logit = -2.8 + 1.0 * inpatient + 0.08 * df["time_in_hospital"]
    df["readmitted"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "<30", rng.choice(["NO", ">30"], n))
    return df
