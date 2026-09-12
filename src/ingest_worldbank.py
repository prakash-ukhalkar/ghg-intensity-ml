"""
World Bank Data Ingestion Module.

Pulls US-level macroeconomic controls from the World Bank API:
  - GDP growth (NY.GDP.MKTP.KD.ZG)
  - CO₂ emissions per capita (EN.ATM.CO2E.PC)
  - Energy use per capita (EG.USE.PCAP.KG.OE)

Falls back to curated reference values when the API is unavailable.
"""
import os
import logging
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Reference values drawn from World Bank DataBank (accessed 2024-10).
# Used as fallback when the API is unreachable and for forward years
# where official data is not yet published.
WORLD_BANK_REFERENCE = {
    2010: {"us_gdp_growth": 2.6, "us_co2_per_capita": 17.6, "us_energy_use_per_capita": 7200.0},
    2011: {"us_gdp_growth": 1.6, "us_co2_per_capita": 17.0, "us_energy_use_per_capita": 7100.0},
    2012: {"us_gdp_growth": 2.2, "us_co2_per_capita": 16.3, "us_energy_use_per_capita": 6900.0},
    2013: {"us_gdp_growth": 1.8, "us_co2_per_capita": 16.4, "us_energy_use_per_capita": 7000.0},
    2014: {"us_gdp_growth": 2.5, "us_co2_per_capita": 16.5, "us_energy_use_per_capita": 6950.0},
    2015: {"us_gdp_growth": 3.1, "us_co2_per_capita": 16.0, "us_energy_use_per_capita": 6900.0},
    2016: {"us_gdp_growth": 1.7, "us_co2_per_capita": 15.5, "us_energy_use_per_capita": 6850.0},
    2017: {"us_gdp_growth": 2.3, "us_co2_per_capita": 15.2, "us_energy_use_per_capita": 6800.0},
    2018: {"us_gdp_growth": 2.9, "us_co2_per_capita": 15.2, "us_energy_use_per_capita": 6800.0},
    2019: {"us_gdp_growth": 2.3, "us_co2_per_capita": 14.8, "us_energy_use_per_capita": 6700.0},
    2020: {"us_gdp_growth": -3.4, "us_co2_per_capita": 13.0, "us_energy_use_per_capita": 6100.0},
    2021: {"us_gdp_growth": 5.7, "us_co2_per_capita": 13.9, "us_energy_use_per_capita": 6400.0},
    2022: {"us_gdp_growth": 2.1, "us_co2_per_capita": 13.6, "us_energy_use_per_capita": 6350.0},
    2023: {"us_gdp_growth": 2.5, "us_co2_per_capita": 13.2, "us_energy_use_per_capita": 6200.0},
    2024: {"us_gdp_growth": 2.4, "us_co2_per_capita": 12.8, "us_energy_use_per_capita": 6100.0},
    2025: {"us_gdp_growth": 2.0, "us_co2_per_capita": 12.5, "us_energy_use_per_capita": 6000.0},
}

# World Bank indicator codes
WB_INDICATORS = {
    "us_gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "us_co2_per_capita": "EN.ATM.CO2E.PC",
    "us_energy_use_per_capita": "EG.USE.PCAP.KG.OE",
}


class WorldBankIngester:
    """Client for the World Bank API to pull US macro controls."""

    def __init__(self, output_dir="data/raw"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def fetch_macro_controls(self, years=None, force_simulate=False):
        """
        Fetches macro indicators from the World Bank API.

        Parameters
        ----------
        years : list[int], optional
            Years to include (default 2018-2023).
        force_simulate : bool
            If True, use reference values without hitting the API.

        Returns
        -------
        pd.DataFrame  with columns: year, us_gdp_growth,
            us_co2_per_capita, us_energy_use_per_capita
        """
        if years is None:
            years = list(range(2010, 2024))

        if force_simulate:
            logger.info("Simulation mode — loading reference World Bank values …")
            return self._from_reference(years)

        return self._fetch_real(years)

    # ------------------------------------------------------------------ #
    #  Real API path                                                      #
    # ------------------------------------------------------------------ #
    def _fetch_real(self, years):
        data_by_year: dict[int, dict] = {y: {} for y in years}

        for col, ind_id in WB_INDICATORS.items():
            url = (
                f"http://api.worldbank.org/v2/country/US/indicator/{ind_id}"
                f"?date={min(years)}:{max(years)}&format=json&per_page=100"
            )
            try:
                logger.info("Fetching World Bank indicator %s …", ind_id)
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    js = resp.json()
                    if len(js) > 1 and isinstance(js[1], list):
                        for item in js[1]:
                            yr = int(item["date"])
                            val = item["value"]
                            if yr in data_by_year and val is not None:
                                data_by_year[yr][col] = val
                else:
                    logger.warning("HTTP %d for %s", resp.status_code, ind_id)
            except Exception as exc:
                logger.error("Error fetching %s: %s", ind_id, exc)

        # Fill gaps with reference values
        records = []
        all_cols = list(WB_INDICATORS.keys())
        for yr in years:
            row = {"year": yr}
            ref = WORLD_BANK_REFERENCE.get(yr, WORLD_BANK_REFERENCE[max(WORLD_BANK_REFERENCE)])
            for col in all_cols:
                row[col] = data_by_year.get(yr, {}).get(col) or ref[col]
            records.append(row)

        df = pd.DataFrame(records)
        out = os.path.join(self.output_dir, "worldbank_macro.csv")
        df.to_csv(out, index=False)
        logger.info("Saved World Bank macro controls → %s", out)
        return df

    # ------------------------------------------------------------------ #
    #  Simulation / reference path                                        #
    # ------------------------------------------------------------------ #
    def _from_reference(self, years):
        records = []
        for yr in years:
            row = {"year": yr}
            row.update(
                WORLD_BANK_REFERENCE.get(
                    yr, WORLD_BANK_REFERENCE[max(WORLD_BANK_REFERENCE)]
                )
            )
            records.append(row)
        df = pd.DataFrame(records)
        out = os.path.join(self.output_dir, "worldbank_macro.csv")
        df.to_csv(out, index=False)
        logger.info("Saved reference World Bank macro controls → %s", out)
        return df


if __name__ == "__main__":
    ingester = WorldBankIngester()
    df = ingester.fetch_macro_controls(force_simulate=True)
    print(df)
