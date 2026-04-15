# DV Assignment 5: Iran Conflict Interactive Dashboard

This project is a Streamlit dashboard that visualizes conflict, market, and environmental indicators using:

- `final_master_dataset.csv`
- `final_dashboard_dataset.csv`
- `iran_conflict_engineered_features.csv`
- `iran_conflict_macro_env_2026.csv`

## Run Locally

1. Create and activate a virtual environment:

	```bash
	python3 -m venv venv
	source venv/bin/activate
	```

2. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

3. Start the app:

	```bash
	streamlit run app.py
	```

## Host Online (Recommended: Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud and sign in with GitHub.
3. Click **Create app** and select:
	- Repository: this repo
	- Branch: `main`
	- Main file path: `app.py`
4. Click **Deploy**.

The app will build using `requirements.txt` automatically.

## Notes

- Do not commit virtual environments (`venv/`, `.venv/`).
- Keep datasets in the repository root so `app.py` can read them with relative paths.
