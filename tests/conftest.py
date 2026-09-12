import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_linked_panel():
    """
    A small synthetic linked panel mimicking EntityResolver.link_datasets()
    output: multiple firms x multiple years, with a reporting/non-reporting
    split and enough variation for Probit/z-score fitting.
    """
    rng = np.random.RandomState(0)
    years = list(range(2018, 2023))
    sectors = ["Manufacturing", "Utilities", "Services"]
    rows = []
    for firm_id in range(40):
        sector = sectors[firm_id % 3]
        # High-emission sectors (Manufacturing/Utilities) are more likely selected
        high_emission = 1 if sector in ("Manufacturing", "Utilities") else 0
        selected = 1 if (high_emission and firm_id % 2 == 0) else 0
        base_assets = rng.uniform(1000, 50000)
        for yr in years:
            assets = base_assets * (1 + 0.05 * (yr - years[0])) * rng.uniform(0.9, 1.1)
            revenue = assets * rng.uniform(0.5, 1.2)
            rows.append({
                "cik": str(firm_id).zfill(10),
                "ticker": f"T{firm_id}",
                "company_name": f"Firm {firm_id}",
                "sector": sector,
                "sic_code": "",
                "year": yr,
                "total_assets": assets,
                "revenue": revenue,
                "net_income": revenue * rng.uniform(0.02, 0.1),
                "operating_income": revenue * rng.uniform(0.05, 0.15),
                "capex": assets * rng.uniform(0.02, 0.06),
                "rd_expense": revenue * rng.uniform(0.0, 0.05),
                "total_debt": assets * rng.uniform(0.2, 0.5),
                "stockholders_equity": assets * rng.uniform(0.3, 0.6),
                "scope1_emissions": (rng.uniform(1000, 100000) if selected else np.nan),
                "co2_non_biogenic": (rng.uniform(900, 95000) if selected else np.nan),
                "n_facilities": rng.randint(1, 5) if selected else 0,
                "primary_naics": 331100,
                "high_emission_naics": high_emission,
                "selected": selected,
                "us_gdp_growth": rng.uniform(1.5, 3.0),
                "us_co2_per_capita": rng.uniform(14, 16),
                "us_energy_use_per_capita": rng.uniform(6500, 7000),
            })
    return pd.DataFrame(rows)
