# Retail Panel Extrapolation and Panel Health Monitoring Simulation Using Real POS Data

## Project Overview
This portfolio project demonstrates how real retail POS panel data can be transformed into market-level sales estimates, then monitored through panel health KPIs and missing retailer risk analysis. This project treats the full available store network as a known universe and evaluates how different real-store sample panels can estimate market-level sales.

## Business Context
The project is inspired by retail panel measurement and POS data quality workflows. It focuses on universe definition, sample panel design, extrapolation, retailer absence scenarios, and KPI monitoring for panel reliability.

## Dataset Used
The project uses the Kaggle dataset "Corporacion Favorita Grocery Sales Forecasting". Place the extracted raw files in `data/raw/` as `.csv` files. The workflow uses real stores, real items, and real observed sales rows only.

## Methodology
The full Favorita store network is treated as the known universe. Sample panels are created by selecting real stores from that universe. Full-universe actual sales are compared with extrapolated estimates from sample panels.

## Universe Definition
The universe workflow identifies active stores using sales and transaction coverage, joins store metadata, creates sales size groups, builds strata from geography and store characteristics, and exports universe summaries.

## Sample Panel Design
The sample design workflow creates random, stratified, large-store-biased, reduced, and optimized panels. Panels are compared by store coverage, geography coverage, sales contribution, category coverage, and top store concentration.

## Extrapolation Methods
The project includes naive store-count extrapolation, stratified extrapolation, historical contribution-weighted extrapolation, and category-level extrapolation. The contribution factor is calibrated from an early baseline window rather than the full evaluation period. Evaluation compares estimated market sales against actual full-universe sales by panel and method.

## Missing Retailer Simulation
Missing retailer scenarios hold out real stores from the panel, including random stores, top-contributing stores, and stores from selected clusters. Imputation methods use similar-store averages, historical trends, and category-level trend adjustment. The analysis separates panel-level recovery from market-level impact after extrapolation.

## Panel Health KPIs
Panel health outputs include active store rate, sample coverage score, category coverage score, top store concentration, panel-specific extrapolation reliability score, panel stability score, overall health score, risk level, and recommended action.

## Dashboard Outputs
Run `streamlit run dashboard/streamlit_app.py` after generating CSV outputs from the notebooks. The dashboard shows universe summaries, panel comparisons, extrapolation performance, weekly and category error, missing retailer impact, and panel health recommendations.

## Limitations
The source data contains observed sales rows only, so absent store-item-date rows are not assumed to be zero sales. Negative `unit_sales` values are treated as product returns. Results depend on the selected date range, product families, and panel design assumptions.

## Relevance to Retail Panel Data Science Roles
This project shows practical data science skills for POS data quality, universe construction, panel sampling, extrapolation, RCA, missing retailer impact analysis, and KPI monitoring in a retail measurement setting.

## Project Structure
```text
config/
data/raw/
data/processed/
data/outputs/
dashboard/
notebooks/
src/
```

## Expected Outputs
The workflow exports CSV files to `data/outputs/`, including universe summaries, sample panel comparisons, extrapolation errors, uncovered strata summaries, missing retailer impact summaries, panel health KPIs, and dashboard summary tables. Generated output files are ignored by Git by default so the repository stays lightweight. Regenerate them locally by running the notebooks in order.
