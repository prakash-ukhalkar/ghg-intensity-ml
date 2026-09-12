"""
Entity Resolution Module.

Links EPA GHGRP parent companies to SEC public-company filings via
two-pass fuzzy matching against the full SEC company index.

Pass 1 — Exact match after aggressive name cleaning.
Pass 2 — Fuzzy match (token-set ratio, threshold ≥ 70) on remaining
          unresolved entities.

Produces an ``epa_sec_mapping.csv`` cross-walk and a final linked panel
that merges EPA emissions, SEC financials, and World Bank macro controls.
"""
import os
import re
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Tokens that frequently differ between EPA and SEC names
_NOISE_TOKENS = re.compile(
    r"\b(inc|corp|co|ltd|llc|llp|lp|plc|sa|nv|group|holdings|"
    r"holding|company|companies|the|of|and|&)\b",
    re.IGNORECASE,
)


def _clean_name(name: str) -> str:
    """Normalise a company name for matching."""
    if not isinstance(name, str):
        return ""
    s = name.upper().strip()
    s = re.sub(r"[^A-Z0-9\s]", " ", s)          # remove punctuation
    s = _NOISE_TOKENS.sub("", s)                  # strip noise tokens
    return " ".join(s.split())                     # collapse whitespace


class EntityResolver:
    """Matches EPA parent companies to SEC public entities."""

    def __init__(self, output_dir: str = "data/interim"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ================================================================== #
    #  Core resolution logic                                              #
    # ================================================================== #
    def resolve_entities(
        self,
        epa_df: pd.DataFrame,
        sec_index: pd.DataFrame,
        fuzzy_threshold: int = 70,
    ) -> pd.DataFrame:
        """
        Matches unique EPA parent-company names to SEC entities.

        Parameters
        ----------
        epa_df : pd.DataFrame
            EPA GHGRP facility data (must contain ``reported_parent``).
        sec_index : pd.DataFrame
            SEC company index (columns: ``cik``, ``ticker``, ``company_name``).
        fuzzy_threshold : int
            Minimum fuzzy score (0-100) for a match (default 70).

        Returns
        -------
        pd.DataFrame
            Cross-walk with columns:
            epa_parent, sec_company_name, cik, ticker, similarity_score, is_resolved
        """
        from rapidfuzz import fuzz, process

        epa_parents = (
            epa_df["reported_parent"]
            .dropna()
            .unique()
            .tolist()
        )
        logger.info("Resolving %d unique EPA parents against %d SEC companies …",
                     len(epa_parents), len(sec_index))

        # Build cleaned lookup for SEC index
        sec_clean = sec_index.copy()
        sec_clean["name_clean"] = sec_clean["company_name"].apply(_clean_name)
        sec_lookup: dict[str, dict] = {}
        for _, row in sec_clean.iterrows():
            nc = row["name_clean"]
            if nc:
                sec_lookup[nc] = {
                    "company_name": row["company_name"],
                    "cik": row["cik"],
                    "ticker": row.get("ticker", ""),
                }

        sec_names_clean = list(sec_lookup.keys())

        results: list[dict] = []
        exact, fuzzy_ok, missed = 0, 0, 0

        for parent in epa_parents:
            pc = _clean_name(parent)
            if not pc:
                results.append(self._miss(parent))
                missed += 1
                continue

            # --- Pass 1: exact ------------------------------------------------
            if pc in sec_lookup:
                info = sec_lookup[pc]
                results.append({
                    "epa_parent": parent,
                    "sec_company_name": info["company_name"],
                    "cik": info["cik"],
                    "ticker": info["ticker"],
                    "similarity_score": 100.0,
                    "is_resolved": True,
                })
                exact += 1
                continue

            # --- Pass 2: fuzzy ------------------------------------------------
            match = process.extractOne(
                pc, sec_names_clean,
                scorer=fuzz.token_set_ratio,
                score_cutoff=fuzzy_threshold,
            )
            if match:
                best_name, score, _ = match
                info = sec_lookup[best_name]
                results.append({
                    "epa_parent": parent,
                    "sec_company_name": info["company_name"],
                    "cik": info["cik"],
                    "ticker": info["ticker"],
                    "similarity_score": float(score),
                    "is_resolved": True,
                })
                fuzzy_ok += 1
            else:
                results.append(self._miss(parent))
                missed += 1

        mapping_df = pd.DataFrame(results)
        out = os.path.join(self.output_dir, "epa_sec_mapping.csv")
        mapping_df.to_csv(out, index=False)
        n_resolved = int(mapping_df["is_resolved"].sum())
        logger.info(
            "Entity resolution complete: %d resolved (%d exact + %d fuzzy), "
            "%d unresolved → %s",
            n_resolved, exact, fuzzy_ok, missed, out,
        )
        return mapping_df

    # ================================================================== #
    #  Dataset linking                                                    #
    # ================================================================== #
    def link_datasets(
        self,
        epa_df: pd.DataFrame,
        sec_df: pd.DataFrame,
        mapping_df: pd.DataFrame,
        wb_df: pd.DataFrame,
        control_ciks: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Produces a linked panel of SEC firms with their EPA emissions (if any)
        and World Bank macro controls.

        Parameters
        ----------
        epa_df : pd.DataFrame
            Facility-level EPA data.
        sec_df : pd.DataFrame
            SEC financials (reporting + control companies).
        mapping_df : pd.DataFrame
            EPA-SEC cross-walk from ``resolve_entities()``.
        wb_df : pd.DataFrame
            World Bank macro controls (year-level).
        control_ciks : list[str], optional
            CIKs of non-reporting control companies.

        Returns
        -------
        pd.DataFrame
            Linked panel with one row per company-year.
        """
        # --- Aggregate EPA to parent-year level ---------------------------
        resolved = mapping_df[mapping_df["is_resolved"]].copy()
        epa_with_cik = epa_df.merge(
            resolved[["epa_parent", "cik"]],
            left_on="reported_parent",
            right_on="epa_parent",
            how="inner",
        )
        epa_agg = (
            epa_with_cik
            .groupby(["cik", "year"], as_index=False)
            .agg(
                scope1_emissions=("total_ghg_emissions", "sum"),
                co2_non_biogenic=("co2_emissions_non_biogenic", "sum"),
                n_facilities=("facility_id", "nunique"),
                primary_naics=("primary_naics_code", "first"),
                high_emission_naics=("high_emission_naics", "max"),
                high_emission_naics_fine=("high_emission_naics_fine", "max"),
            )
        )

        # --- Merge SEC financials with EPA aggregates ---------------------
        linked = sec_df.merge(
            epa_agg, on=["cik", "year"], how="left",
        )

        # selected = 1 if company reports to EPA in that year
        linked["selected"] = (~linked["scope1_emissions"].isna()).astype(int)

        # For non-reporting companies, carry forward the high_emission_naics
        # from their sector (we infer from SEC sector for controls).
        if "high_emission_naics" not in linked.columns:
            linked["high_emission_naics"] = 0
        linked["high_emission_naics"] = linked["high_emission_naics"].fillna(0).astype(int)

        if "high_emission_naics_fine" not in linked.columns:
            linked["high_emission_naics_fine"] = 0
        linked["high_emission_naics_fine"] = linked["high_emission_naics_fine"].fillna(0).astype(int)

        # If control CIKs provided, flag them
        if control_ciks:
            control_set = set(str(c).zfill(10) for c in control_ciks)
            # For controls in heavy-industry sectors, set high_emission_naics = 1
            heavy = {"Mining", "Manufacturing", "Utilities", "Transportation"}
            mask = (
                linked["cik"].isin(control_set)
                & linked["sector"].isin(heavy)
            )
            linked.loc[mask, "high_emission_naics"] = 1

        # --- Merge World Bank macro controls ------------------------------
        linked = linked.merge(wb_df, on="year", how="left")

        out = os.path.join(self.output_dir, "linked_panel.csv")
        linked.to_csv(out, index=False)

        n_reporting = int(linked["selected"].sum())
        n_total = len(linked)
        n_firms = linked["cik"].nunique()
        logger.info(
            "Linked panel: %d firm-years (%d unique firms), "
            "%d reporting, %d non-reporting → %s",
            n_total, n_firms, n_reporting, n_total - n_reporting, out,
        )
        return linked

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _miss(parent: str) -> dict:
        return {
            "epa_parent": parent,
            "sec_company_name": "",
            "cik": "",
            "ticker": "",
            "similarity_score": 0.0,
            "is_resolved": False,
        }


if __name__ == "__main__":
    print("Entity resolver module — import and call resolve_entities().")
