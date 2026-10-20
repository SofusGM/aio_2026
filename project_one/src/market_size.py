"""Estimate annual household counts from census data and add them to the car-market data."""


import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import parts
sns.set_theme(style='whitegrid', context='notebook')



DATA = parts.DATA
RES = parts.RES / 'market size'
RES.mkdir(parents=True, exist_ok=True)


# %% Read in data
cars = pd.read_csv(DATA / 'cars.csv')
cars = cars.drop(columns="households_pred", errors="ignore")

lbl_vars = pd.read_csv(DATA / 'labels_variables.csv', index_col=0)['label']
lbl_vals = pd.read_stata(DATA / 'cars.dta', iterator=True).value_labels()



### Plot population over time ###

cars["year"] = 1900 + cars["ye"]
cars["market"] = cars["ma"].map(lbl_vals["market"])

# Keep one population observation per market-year
pop = cars.drop_duplicates(["market", "year"])

fig, ax = plt.subplots(figsize=(9, 6))
sns.lineplot(data=pop, x="year", y="pop", hue="market", errorbar=None, ax=ax)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1_000_000:g}"))

ax.set(xlim=(1970, 1999), xlabel="Year", ylabel="Population (millions)")
ax.set_xticks(range(1970, 2000, 5))
ax.legend(title="Market", frameon=False)

fig.tight_layout()


fig.savefig(RES / "population_by_market.png", dpi=200, bbox_inches="tight")


### load census data for household size
household = pd.read_csv(DATA / 'household_census.csv', na_values=[".."])
household = household.dropna(subset=["ma", "ye", "households"])

# linear trend through the census points (Germany split at reunification)
models = {}
predictions = []

for market, market_data in household.groupby("ma"):
    if market == "Germany":
        periods = [
            ("1970–1990", market_data[market_data["ye"] <= 1990],
             np.arange(1970, 1991)),
            ("1991–2000", market_data[market_data["ye"] >= 1991],
             np.arange(1991, 2001)),
        ]
    else:
        periods = [
            ("full period", market_data, np.arange(1970, 2001))
        ]

    for period, data, years in periods:
        X = sm.add_constant(data[["ye"]], has_constant="add")
        model = sm.OLS(data["households"], X).fit()
        models[(market, period)] = model

        prediction_data = pd.DataFrame({"ye": years})
        X_prediction = sm.add_constant(
            prediction_data[["ye"]], has_constant="add"
        )

        prediction_data["ma"] = market
        prediction_data["households_pred"] = model.predict(X_prediction)
        predictions.append(prediction_data)

predicted_households = pd.concat(predictions, ignore_index=True)


# plot census observations and regression predictions
markets = sorted(household["ma"].unique())
palette = dict(zip(markets, sns.color_palette("tab10", n_colors=len(markets))))

fig, ax = plt.subplots(figsize=(9, 6))

# Census observations
sns.scatterplot(
    data=household,
    x="ye",
    y="households",
    hue="ma",
    palette=palette,
    s=80,
    ax=ax
)

# Regression predictions, including Germany's separate pre- and post-1990 fits
for market in markets:
    fitted = predicted_households[predicted_households["ma"] == market].sort_values("ye")
    ax.plot(
        fitted["ye"],
        fitted["households_pred"],
        color=palette[market],
        linewidth=2
    )

ax.set(xlim=(1970, 2000), xlabel="Year", ylabel="Households (millions)")
ax.set_xticks(range(1970, 2001, 5))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1_000_000:g}"))
ax.legend(title="Country", frameon=False)
fig.tight_layout()

fig.savefig(RES / "household_regressions.png", dpi=200, bbox_inches="tight")

plt.show()

### save predicted households as variable in cars df ###

# Convert predicted years: 1970–1999 -> 70–99
pred = predicted_households.copy()
pred["ye"] = pred["ye"].astype(int) - 1900
pred = pred[pred["ye"].between(70, 99)]


# Match the census market names to the car data's market codes
market_codes = {name: code for code, name in lbl_vals["market"].items()}
pred["ma"] = pred["ma"].map(market_codes)

# Add predicted household counts to each car row
cars = cars[cars["ye"].between(70, 99)]
cars = cars.merge(pred[["ma", "ye", "households_pred"]], on=["ma", "ye"], how="left", validate="many_to_one"
)

# Overwrite cars.csv
cars.to_csv(DATA / "cars.csv", index=False)



### Plot predicted households against the population in the car data ###
markets = sorted(household["ma"].dropna().unique())

fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
axes = axes.ravel()

for ax, market in zip(axes, markets):
    census = household[household["ma"] == market].dropna(
        subset=["ye", "households"]
    )

    # Select this market's population data
    car_pop = pop[pop["market"] == market].sort_values("year")

    population_line, = ax.plot(
        car_pop["year"],
        car_pop["pop"] / 1_000_000,
        color="tab:blue",
        label="Population"
    )

    if market == "Germany":
        periods = [
            (
                census[census["ye"] <= 1990],
                np.arange(1970, 1991),
                "Predicted households 1970–1990",
                "--"
            ),
            (
                census[census["ye"] >= 1991],
                np.arange(1991, 2001),
                "Predicted households 1991–2000",
                ":"
            ),
        ]
    else:
        periods = [
            (
                census,
                np.arange(1970, 2001),
                "Predicted households",
                "--"
            )
        ]

    regression_lines = []

    for data, years, label, linestyle in periods:
        if len(data) < 2:
            continue

        model = sm.OLS(
            data["households"],
            sm.add_constant(data[["ye"]], has_constant="add")
        ).fit()

        prediction_data = pd.DataFrame({"ye": years})
        predictions = model.predict(
            sm.add_constant(prediction_data[["ye"]], has_constant="add")
        )

        line, = ax.plot(
            years,
            predictions / 1_000_000,
            color="black",
            linestyle=linestyle,
            label=label
        )
        regression_lines.append(line)

    ax.set_title(market)
    ax.set_xlim(1970, 2000)
    ax.set_xticks(range(1970, 2001, 5))
    ax.set_xlabel("Year")
    ax.set_ylabel("People / households (millions)")

    lines = regression_lines + [population_line]
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8)

for ax in axes[len(markets):]:
    ax.set_visible(False)

fig.tight_layout()
fig.savefig(RES / "population_and_household_regressions.png", dpi=200, bbox_inches="tight")

plt.show()