"""
EPA GHGRP Data Ingestion Module.

Downloads and processes the full EPA Greenhouse Gas Reporting Program (GHGRP)
facility-level emissions data. Reports only Scope 1 direct emissions (the only
scope reported under GHGRP). Adds NAICS-based sector classification and a
high-emission NAICS flag used as the Heckman exclusion restriction variable.

Data source: https://www.epa.gov/ghgreporting/data-sets
"""
import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NAICS 2-digit code → broad sector label
# ---------------------------------------------------------------------------
NAICS_TO_SECTOR = {
    11: "Agriculture", 21: "Mining", 22: "Utilities",
    23: "Construction",
    31: "Manufacturing", 32: "Manufacturing", 33: "Manufacturing",
    42: "Wholesale",
    44: "Retail", 45: "Retail",
    48: "Transportation", 49: "Transportation",
    51: "Information", 52: "Finance", 53: "Real Estate",
    54: "Professional Services", 55: "Management", 56: "Administrative",
    61: "Education", 62: "Healthcare", 71: "Arts/Entertainment",
    72: "Accommodation/Food", 81: "Other Services", 92: "Public Administration",
}

# NAICS 2-digit codes whose facilities structurally exceed the EPA 25 000
# metric-ton CO2e reporting threshold.  Used as the Heckman exclusion
# restriction: predicts *selection* into GHGRP but not emissions *intensity*.
HIGH_EMISSION_NAICS_2D = {21, 22, 31, 32, 33, 48, 49}

# Finer, 6-digit NAICS codes for specific GHGRP-mandatory source categories
# (fossil-fuel electricity generation, petroleum refining, cement, iron and
# steel, petrochemicals). A SECOND, more granular exclusion-restriction
# candidate: less collinear with the broad `sector` categorical (derived
# from 2-digit SIC) than HIGH_EMISSION_NAICS_2D, used only for the
# identification-strength robustness check in
# ``Evaluator.run_robustness_checks`` — NOT for the main Heckman fit, which
# keeps using the 2-digit flag throughout.
HIGH_EMISSION_NAICS_6D_FINE = {
    221112, 221114, 221121, 221122,  # fossil-fuel electricity generation
    324110,                          # petroleum refining
    327310,                          # cement manufacturing
    331110,                          # iron and steel mills
    325110,                          # petrochemical manufacturing
}


class EPAIngester:
    """
    Processes all EPA GHGRP facility-level direct (Scope 1) emissions.

    Real-data path
    ──────────────
    Downloads bulk GHGRP Excel workbooks and the parent-company cross-walk,
    producing a panel of ~7 000 facilities × 6 years.

    Simulation path
    ───────────────
    Generates ~300 parent companies and ~1 200 facilities with realistic
    sector-differentiated emissions distributions.
    """

    def __init__(self, output_dir="data/raw"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #
    def fetch_epa_emissions(self, years=None, force_simulate=False):
        """
        Fetches EPA GHGRP emissions for ALL reporting facilities.

        Parameters
        ----------
        years : list[int], optional
            Reporting years to include (default 2018-2023).
        force_simulate : bool
            If True, bypass downloads and generate synthetic data at scale.

        Returns
        -------
        pd.DataFrame  with columns:
            facility_id, facility_name, reported_parent, year, state,
            total_ghg_emissions, co2_emissions_non_biogenic,
            primary_naics_code, naics_2digit, naics_sector,
            high_emission_naics
        """
        if years is None:
            years = list(range(2010, 2024))

        if force_simulate:
            logger.info("Simulation mode — generating large-scale EPA emissions …")
            return self._generate_simulated_epa(years)

        return self._fetch_real_epa(years)

    # ------------------------------------------------------------------ #
    #  Real data path                                                     #
    # ------------------------------------------------------------------ #
    def _fetch_real_epa(self, years):
        import zipfile
        import requests

        parent_url = (
            "https://www.epa.gov/system/files/other-files/"
            "2024-10/ghgp_data_parent_company.xlsb"
        )
        zip_url = (
            "https://www.epa.gov/system/files/other-files/"
            "2024-10/2023_data_summary_spreadsheets.zip"
        )

        xlsb_path = os.path.join(self.output_dir, "parent_company.xlsb")
        zip_path = os.path.join(self.output_dir, "data_summaries.zip")

        # Download if not cached
        for filepath, url in [(xlsb_path, parent_url), (zip_path, zip_url)]:
            if not os.path.exists(filepath):
                logger.info("Downloading %s …", os.path.basename(filepath))
                resp = requests.get(url, timeout=120, stream=True)
                resp.raise_for_status()
                with open(filepath, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=8192):
                        fh.write(chunk)
                logger.info("Downloaded %s", os.path.basename(filepath))

        all_dfs = []
        for year in years:
            logger.info("Processing EPA GHGRP data for %d …", year)
            xlsx_fname = f"ghgp_data_{year}.xlsx"
            xlsx_path = os.path.join(self.output_dir, xlsx_fname)
            if not os.path.exists(xlsx_path):
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extract(xlsx_fname, self.output_dir)

            # GHGRP renamed this sheet from "Direct Emitters" (2010-2017) to
            # "Direct Point Emitters" (2018+); columns are unchanged across
            # the rename, so no other year-dependent handling is needed.
            sheet_name = "Direct Point Emitters" if year >= 2018 else "Direct Emitters"
            df_em = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=3)
            df_em.columns = df_em.columns.str.strip()

            df_pc = pd.read_excel(
                xlsb_path, sheet_name=str(year), engine="pyxlsb"
            )
            df_pc.columns = df_pc.columns.str.strip()

            # Facility emissions
            df_em_sub = df_em[
                [
                    "Facility Id",
                    "Facility Name",
                    "State",
                    "Total reported direct emissions",
                    "CO2 emissions (non-biogenic)",
                    "Primary NAICS Code",
                ]
            ].copy()

            # Parent-company crosswalk
            df_pc_sub = (
                df_pc[["GHGRP FACILITY ID", "PARENT COMPANY NAME"]]
                .copy()
                .rename(
                    columns={
                        "GHGRP FACILITY ID": "Facility Id",
                        "PARENT COMPANY NAME": "reported_parent",
                    }
                )
                .drop_duplicates(subset=["Facility Id"])
            )

            df_m = pd.merge(df_em_sub, df_pc_sub, on="Facility Id", how="left")
            df_m = df_m.rename(
                columns={
                    "Facility Id": "facility_id",
                    "Facility Name": "facility_name",
                    "State": "state",
                    "Total reported direct emissions": "total_ghg_emissions",
                    "CO2 emissions (non-biogenic)": "co2_emissions_non_biogenic",
                    "Primary NAICS Code": "primary_naics_code",
                }
            )
            df_m["year"] = year

            # Derive NAICS sector and exclusion restriction flag
            df_m["naics_2digit"] = pd.to_numeric(
                df_m["primary_naics_code"].astype(str).str[:2], errors="coerce"
            )
            df_m["naics_sector"] = (
                df_m["naics_2digit"].map(NAICS_TO_SECTOR).fillna("Other")
            )
            df_m["high_emission_naics"] = (
                df_m["naics_2digit"].isin(HIGH_EMISSION_NAICS_2D).astype(int)
            )
            df_m["high_emission_naics_fine"] = (
                pd.to_numeric(df_m["primary_naics_code"], errors="coerce")
                .isin(HIGH_EMISSION_NAICS_6D_FINE)
                .astype(int)
            )

            cols = [
                "facility_id", "facility_name", "reported_parent",
                "year", "state",
                "total_ghg_emissions", "co2_emissions_non_biogenic",
                "primary_naics_code", "naics_2digit", "naics_sector",
                "high_emission_naics", "high_emission_naics_fine",
            ]
            all_dfs.append(df_m[cols])

        df_all = pd.concat(all_dfs, ignore_index=True)
        out = os.path.join(self.output_dir, "epa_ghgrp_facilities.csv")
        df_all.to_csv(out, index=False)
        logger.info(
            "Saved %d facility records (%d unique parents) → %s",
            len(df_all), df_all["reported_parent"].nunique(), out,
        )
        return df_all

    # ------------------------------------------------------------------ #
    #  Simulation path                                                    #
    # ------------------------------------------------------------------ #
    def _generate_simulated_epa(self, years):
        """
        Generates large-scale synthetic EPA facility data that mirrors the
        real GHGRP structure: ~300 parent companies, ~1 200 facilities,
        sector-differentiated emissions, and a mild decarbonisation trend.
        """
        np.random.seed(101)
        records: list[dict] = []

        sector_configs = {
            "Utilities":       {"n": 80,  "fac": (3, 15), "em": (500_000, 10_000_000), "naics": [221112, 221114, 221118, 221121, 221122]},
            "Mining":          {"n": 40,  "fac": (2, 8),  "em": (100_000, 3_000_000),  "naics": [211120, 212210, 212220, 212230, 212310]},
            "Manufacturing":   {"n": 100, "fac": (2, 10), "em": (50_000,  2_000_000),  "naics": [324110, 325110, 325180, 327310, 331110]},
            "Transportation":  {"n": 30,  "fac": (2, 6),  "em": (30_000,  500_000),    "naics": [486110, 486210, 481111, 482111, 484110]},
            "Agriculture":     {"n": 20,  "fac": (1, 4),  "em": (25_000,  300_000),    "naics": [111140, 111998, 112120, 115310, 311221]},
            "Other":           {"n": 30,  "fac": (1, 3),  "em": (25_000,  200_000),    "naics": [518210, 562111, 562212, 423510, 541380]},
        }

        states = [
            "TX", "CA", "OH", "PA", "IL", "FL", "NY", "NC",
            "LA", "WV", "WY", "KY", "IN", "MI", "AL", "GA",
        ]
        fac_id = 100_001
        parent_id = 1

        for sector, cfg in sector_configs.items():
            for _ in range(cfg["n"]):
                parent_name = f"SimCo-{parent_id:04d} {sector} Corp"
                parent_id += 1
                n_fac = np.random.randint(*cfg["fac"])
                naics = int(np.random.choice(cfg["naics"]))
                naics_2d = int(str(naics)[:2])

                for fi in range(n_fac):
                    fid = fac_id; fac_id += 1
                    st = np.random.choice(states)

                    for yr in years:
                        base = np.random.uniform(*cfg["em"])
                        decarb = 1.0 - (yr - years[0]) * np.random.uniform(0.01, 0.04)
                        em = max(1_000.0, base * decarb + np.random.normal(0, base * 0.08))
                        co2_nb = em * np.random.uniform(0.85, 0.98)

                        records.append({
                            "facility_id": fid,
                            "facility_name": f"{parent_name} Fac-{chr(65 + fi)}",
                            "reported_parent": parent_name,
                            "year": yr,
                            "state": st,
                            "total_ghg_emissions": round(em, 2),
                            "co2_emissions_non_biogenic": round(co2_nb, 2),
                            "primary_naics_code": naics,
                            "naics_2digit": naics_2d,
                            "naics_sector": NAICS_TO_SECTOR.get(naics_2d, "Other"),
                            "high_emission_naics": 1 if naics_2d in HIGH_EMISSION_NAICS_2D else 0,
                            "high_emission_naics_fine": 1 if naics in HIGH_EMISSION_NAICS_6D_FINE else 0,
                        })

        df = pd.DataFrame(records)
        out = os.path.join(self.output_dir, "epa_ghgrp_facilities.csv")
        df.to_csv(out, index=False)
        logger.info(
            "Saved %d simulated facility records (%d parents) → %s",
            len(df), df["reported_parent"].nunique(), out,
        )
        return df


if __name__ == "__main__":
    ingester = EPAIngester()
    df = ingester.fetch_epa_emissions()
    print(f"Shape: {df.shape},  Unique parents: {df['reported_parent'].nunique()}")
    print(df.head())
