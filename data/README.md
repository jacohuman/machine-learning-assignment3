# Data provenance

All five of the datasets are public-domain datasets in statsmodels 0.14.6.
`src/datasets.py` exports them without network access.

- **Nile.csv**: annual Nile discharge at Aswan, 1871–1970, in 10^8 cubic metres.
  https://www.statsmodels.org/stable/datasets/generated/nile.html
- **Sunspots.csv**: yearly sunspot activity, 1700–2008.
  https://www.statsmodels.org/stable/datasets/generated/sunspots.html
- **CO2.csv**: monthly averages of available weekly Mauna Loa CO2 measurements,
  March 1958–December 2001, ppm. Fully missing months remain missing in the CSV.
  https://www.statsmodels.org/stable/datasets/generated/co2.html
- **ElNino.csv**: monthly Niño 1+2 sea surface temperature, 1950–2010, Celsius.
  The 12 month columns are flattened in calendar order.
  https://www.statsmodels.org/stable/datasets/generated/elnino.html
- **GDP.csv**: quarterly US real GDP, 1959Q1–2009Q3, billions of chained 2005 USD,
  seasonally adjusted annual rate; other macrodata columns are not used.
  https://www.statsmodels.org/stable/datasets/generated/macrodata.html
