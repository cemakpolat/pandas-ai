"""
Sample dataset generators for testing and demonstration.

These functions are intentionally separated from DataManager so that the
core lifecycle class does not carry fixture/demo logic.
"""

import numpy as np
import pandas as pd


def create_sales_data() -> pd.DataFrame:
	"""Create sample sales data (100 rows)."""
	np.random.seed(42)
	dates = pd.date_range('2024-01-01', periods=100)

	return pd.DataFrame({
		'date': dates,
		'product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Monitor'], 100),
		'quantity': np.random.randint(1, 20, 100),
		'price': np.random.uniform(100, 2000, 100),
		'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
		'salesperson': np.random.choice(['Alice', 'Bob', 'Charlie', 'Diana'], 100),
	})


def create_weather_data() -> pd.DataFrame:
	"""Create sample weather data (365 rows)."""
	np.random.seed(42)
	dates = pd.date_range('2024-01-01', periods=365)

	return pd.DataFrame({
		'date': dates,
		'temperature': np.random.uniform(-5, 35, 365),
		'humidity': np.random.uniform(30, 100, 365),
		'pressure': np.random.uniform(950, 1050, 365),
		'rainfall': np.random.exponential(5, 365),
		'city': np.random.choice(['New York', 'London', 'Tokyo', 'Sydney'], 365),
	})


def create_stocks_data() -> pd.DataFrame:
	"""Create sample stock market data (252 rows)."""
	np.random.seed(42)
	dates = pd.date_range('2024-01-01', periods=252)

	base_price = 100
	prices = [base_price]
	for _ in range(251):
		change = np.random.normal(0, 2)
		prices.append(max(prices[-1] + change, 10))

	return pd.DataFrame({
		'date': dates,
		'ticker': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN'], 252),
		'open': prices,
		'high': np.array(prices) + np.abs(np.random.normal(0, 1, 252)),
		'low': np.array(prices) - np.abs(np.random.normal(0, 1, 252)),
		'close': np.array(prices) + np.random.normal(0, 0.5, 252),
		'volume': np.random.randint(1000000, 10000000, 252),
	})


def create_analytics_data() -> pd.DataFrame:
	"""Create sample web analytics data (90 rows)."""
	np.random.seed(42)
	dates = pd.date_range('2024-01-01', periods=90)

	return pd.DataFrame({
		'date': dates,
		'page': np.random.choice(['/home', '/about', '/products', '/blog', '/contact'], 90),
		'visitors': np.random.randint(100, 5000, 90),
		'pageviews': np.random.randint(200, 10000, 90),
		'bounce_rate': np.random.uniform(20, 80, 90),
		'avg_session_duration': np.random.uniform(30, 600, 90),
		'conversion_rate': np.random.uniform(0.5, 5, 90),
	})


def create_ecommerce_data() -> pd.DataFrame:
	"""Create realistic e-commerce order data (2000 rows).

	Features deliberately encoded:
	- Seasonal Q4 sales surge (Oct-Dec ~40% higher revenue)
	- Category-specific price/discount/rating profiles
	- Higher discounts → lower ratings (negative correlation)
	- Mobile channel skews toward lower-value, higher-frequency orders
	"""
	np.random.seed(7)
	n = 2_000
	dates = pd.to_datetime(
		np.random.choice(pd.date_range('2023-01-01', '2024-12-31'), n, replace=True)
	)
	months = dates.month

	categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
	countries  = ['USA', 'Germany', 'UK', 'France', 'Japan']
	channels   = ['web', 'mobile', 'app']
	segments   = ['new', 'returning', 'vip']

	cat_cfg = {
		'Electronics': (120, 1200, 12, 6,  3.9),
		'Clothing':    ( 20,  250, 20, 8,  4.1),
		'Books':       (  8,   70,  5, 3,  4.4),
		'Home':        ( 30,  500, 15, 7,  4.0),
		'Sports':      ( 25,  350, 18, 8,  4.2),
	}
	cat_weights = [0.30, 0.28, 0.12, 0.18, 0.12]
	cats = np.random.choice(categories, n, p=cat_weights)

	prices, discounts, base_ratings = [], [], []
	for cat in cats:
		lo, hi, dm, ds, rm = cat_cfg[cat]
		prices.append(np.random.uniform(lo, hi))
		discounts.append(np.clip(np.random.normal(dm, ds), 0, 50))
		base_ratings.append(rm)

	prices       = np.array(prices)
	discounts    = np.array(discounts)
	base_ratings = np.array(base_ratings)

	seasonal_mult = np.where(
		months.isin([10, 11, 12]),
		np.random.uniform(1.15, 1.45, n),
		np.random.uniform(0.85, 1.05, n),
	)
	prices = (prices * seasonal_mult).round(2)

	rating_noise = np.random.normal(0, 0.3, n)
	ratings = np.clip(
		base_ratings - 0.04 * (discounts - 15) + rating_noise, 1.0, 5.0
	).round(1)

	channel_arr = np.random.choice(channels, n, p=[0.45, 0.35, 0.20])
	units = np.where(
		channel_arr == 'mobile',
		np.random.randint(1, 3, n),
		np.random.randint(1, 6, n),
	)

	return_prob = np.clip(0.05 + 0.004 * discounts - 0.02 * (ratings - 3), 0.02, 0.40)
	returned = np.random.binomial(1, return_prob)
	customer_segment = np.random.choice(segments, n, p=[0.40, 0.45, 0.15])

	return pd.DataFrame({
		'date':             dates,
		'month':            months,
		'quarter':          ((months - 1) // 3 + 1),
		'category':         cats,
		'country':          np.random.choice(countries, n),
		'channel':          channel_arr,
		'customer_segment': customer_segment,
		'units':            units,
		'order_value':      prices,
		'discount_pct':     discounts.round(1),
		'final_value':      (prices * units * (1 - discounts / 100)).round(2),
		'rating':           ratings,
		'returned':         returned,
	})


def create_health_data() -> pd.DataFrame:
	"""Create realistic patient health metrics (1200 rows)."""
	np.random.seed(13)
	n = 1_200
	gender   = np.random.choice(['M', 'F'], n, p=[0.48, 0.52])
	age      = np.random.randint(18, 85, n)
	smoking  = np.random.choice([0, 1], n, p=[0.72, 0.28])
	exercise = np.random.randint(0, 8, n)

	bmi = np.clip(
		np.random.normal(27, 5, n) - exercise * 0.4 + smoking * 1.5 + np.where(age > 50, 1.5, 0),
		14, 50,
	).round(1)

	bp = np.clip(
		100 + age * 0.50 + (bmi - 25) * 0.80 + smoking * 8 - exercise * 1.5
		+ np.random.normal(0, 7, n),
		65, 210,
	).round(0).astype(int)

	chol = np.clip(
		150 + age * 0.70 + (bmi - 25) * 1.2 + smoking * 15 - exercise * 2
		+ np.random.normal(0, 20, n),
		100, 380,
	).round(0).astype(int)

	glucose = np.clip(
		70 + (bmi - 25) * 1.5 + age * 0.30 + smoking * 10 + np.random.normal(0, 12, n),
		55, 350,
	).round(0).astype(int)

	vo2max = np.clip(
		60 - age * 0.30 - (bmi - 22) * 0.40 + exercise * 1.8 + np.random.normal(0, 3, n),
		15, 75,
	).round(1)

	risk_score = (
		(age > 55).astype(int)
		+ (bmi > 30).astype(int)
		+ (bp > 140).astype(int)
		+ (glucose > 125).astype(int)
		+ smoking
		+ (exercise < 2).astype(int)
	)
	outcome   = np.where(risk_score >= 3, 'at_risk', 'healthy')
	age_group = pd.cut(
		age,
		bins=[17, 30, 45, 60, 85],
		labels=['18-30', '31-45', '46-60', '61+'],
	).astype(str)

	return pd.DataFrame({
		'patient_id':     np.arange(1, n + 1),
		'age':            age,
		'age_group':      age_group,
		'gender':         gender,
		'bmi':            bmi,
		'blood_pressure': bp,
		'cholesterol':    chol,
		'glucose':        glucose,
		'vo2max':         vo2max,
		'smoking':        smoking,
		'exercise_days':  exercise,
		'risk_score':     risk_score,
		'outcome':        outcome,
	})


def create_energy_data() -> pd.DataFrame:
	"""Create realistic 3-year daily energy data (sources × days ≈ 5475 rows)."""
	np.random.seed(21)
	dates   = pd.date_range('2022-01-01', '2024-12-31')
	sources = ['Solar', 'Wind', 'Gas', 'Nuclear', 'Coal']
	regions = ['North', 'South', 'East', 'West']

	co2_factor = {'Solar': 0.0, 'Wind': 0.0, 'Gas': 0.49, 'Nuclear': 0.012, 'Coal': 0.82}
	year_mult  = {2022: 1.00, 2023: 0.97, 2024: 0.94}

	rows = []
	for date in dates:
		doy           = date.day_of_year
		year          = date.year
		solar_season  = max(0.0, np.sin(np.pi * (doy - 80) / 365))
		wind_season   = 1.0 - solar_season
		demand_season = 0.7 + 0.3 * (abs(np.sin(np.pi * doy / 365)) ** 0.5)

		renew_gwh_approx = (20 + 25 * solar_season) + (30 + 20 * wind_season)
		total_approx     = renew_gwh_approx + 80 + 60 + 50
		renew_share      = renew_gwh_approx / total_approx
		spot_price       = round(
			np.clip(80 - 55 * renew_share + np.random.normal(0, 5), 10, 200), 2
		)

		for src in sources:
			if src == 'Solar':
				base     = 20 + 25 * solar_season
				cap_mean = 20 + 65 * solar_season
			elif src == 'Wind':
				base     = 30 + 20 * wind_season
				cap_mean = 30 + 55 * wind_season
			elif src == 'Gas':
				base     = 60 + 30 * demand_season
				cap_mean = 55 + 25 * demand_season
			elif src == 'Nuclear':
				maint_dip = np.random.choice([0, 1], p=[0.90, 0.10])
				base      = 58 * (1 - 0.8 * maint_dip)
				cap_mean  = 85 - 60 * maint_dip
			else:
				base     = 50 * year_mult[year]
				cap_mean = 60 * year_mult[year]

			noise       = np.random.normal(0, base * 0.08 + 1)
			production  = max(0.0, base + noise)
			consumption = production * np.random.uniform(0.82, 0.99)
			co2         = production * co2_factor[src]
			capacity    = np.clip(np.random.normal(cap_mean, 6), 5, 100)
			cost_usd    = round(production * spot_price, 2)

			rows.append({
				'date':            date,
				'year':            year,
				'month':           date.month,
				'season':          ['Winter', 'Winter', 'Spring', 'Spring', 'Spring',
				                    'Summer', 'Summer', 'Summer', 'Autumn', 'Autumn',
				                    'Autumn', 'Winter'][date.month - 1],
				'source':          src,
				'region':          np.random.choice(regions),
				'production_gwh':  round(production, 2),
				'consumption_gwh': round(consumption, 2),
				'co2_tonnes':      round(co2, 3),
				'capacity_pct':    round(capacity, 1),
				'spot_price_mwh':  spot_price,
				'cost_usd_k':      cost_usd,
			})
	return pd.DataFrame(rows)
