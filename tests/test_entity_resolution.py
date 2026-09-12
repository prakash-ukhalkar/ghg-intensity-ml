import pandas as pd

from src.entity_resolution import EntityResolver


def test_high_emission_naics_fine_propagates_through_linking(tmp_path):
    """
    The fine (6-digit) exclusion-restriction candidate must survive
    EntityResolver.link_datasets' aggregation and merge, with the same
    "max within firm-year, fill 0 for non-reporting" behavior as the
    existing broad (2-digit) flag.

    Uses tmp_path for output_dir so this test never overwrites the real
    project's data/interim/linked_panel.csv.
    """
    epa_df = pd.DataFrame({
        "facility_id": [1, 2],
        "reported_parent": ["Acme Corp", "Acme Corp"],
        "year": [2020, 2020],
        "total_ghg_emissions": [1000.0, 2000.0],
        "co2_emissions_non_biogenic": [900.0, 1900.0],
        "primary_naics_code": [324110, 221112],
        "high_emission_naics": [1, 1],
        "high_emission_naics_fine": [1, 1],
    })
    mapping_df = pd.DataFrame({
        "epa_parent": ["Acme Corp"],
        "sec_company_name": ["Acme Corp"],
        "cik": ["0000000001"],
        "ticker": ["ACME"],
        "similarity_score": [100.0],
        "is_resolved": [True],
    })
    sec_df = pd.DataFrame({
        "cik": ["0000000001", "0000000002"],
        "ticker": ["ACME", "OTHR"],
        "company_name": ["Acme Corp", "Other Inc"],
        "sector": ["Manufacturing", "Retail"],
        "sic_code": ["", ""],
        "year": [2020, 2020],
        "total_assets": [1_000_000.0, 500_000.0],
        "revenue": [2_000_000.0, 1_000_000.0],
        "net_income": [100_000.0, 50_000.0],
        "operating_income": [150_000.0, 60_000.0],
        "capex": [50_000.0, 20_000.0],
        "rd_expense": [10_000.0, 5_000.0],
        "total_debt": [300_000.0, 100_000.0],
        "stockholders_equity": [700_000.0, 400_000.0],
    })
    wb_df = pd.DataFrame({
        "year": [2020],
        "us_gdp_growth": [2.1],
        "us_co2_per_capita": [14.5],
        "us_energy_use_per_capita": [6800.0],
    })

    resolver = EntityResolver(output_dir=str(tmp_path))
    linked = resolver.link_datasets(epa_df, sec_df, mapping_df, wb_df, control_ciks=["0000000002"])

    acme_row = linked[linked["cik"] == "0000000001"].iloc[0]
    other_row = linked[linked["cik"] == "0000000002"].iloc[0]

    assert acme_row["high_emission_naics_fine"] == 1
    assert other_row["high_emission_naics_fine"] == 0
    assert acme_row["selected"] == 1
    assert other_row["selected"] == 0
