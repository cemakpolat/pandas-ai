"""
api.py — FastAPI REST wrapper for PyChartAI.

Provides ``/chat``, ``/chart``, and ``/profile`` endpoints that accept
data + query via JSON and return analysis results.

Launch::

    uvicorn pychartai_core.api:app --reload

Or from code::

    from pychartai_core.api import create_app
    app = create_app()
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import Any, Dict, List, Optional

import pandas as pd

try:
	from fastapi import FastAPI, HTTPException, UploadFile, File, Form
	from fastapi.responses import FileResponse, JSONResponse
	from pydantic import BaseModel
except ImportError:
	raise ImportError(
		'FastAPI is required for the REST API. '
		'Install it with: pip install fastapi uvicorn'
	)


# ---- Request / Response models ----

class ChatRequest(BaseModel):
	"""Request body for /chat endpoint."""
	query: str
	data: Optional[List[Dict[str, Any]]] = None
	chart_library: str = 'seaborn'
	explain: bool = False


class ChatResponse(BaseModel):
	"""Response from /chat endpoint."""
	result: str
	is_chart: bool = False
	chart_path: Optional[str] = None


class ProfileResponse(BaseModel):
	"""Response from /profile endpoint."""
	n_rows: int
	n_columns: int
	memory_usage_mb: float
	dtypes: Dict[str, int]
	duplicates: int
	constant_columns: List[str]
	high_cardinality: List[str]
	summary: str


def create_app(llm=None) -> FastAPI:
	"""Create a configured FastAPI application.

	Args:
		llm: Optional LLM instance to use.  Falls back to global config.
	"""
	app = FastAPI(
		title='PyChartAI API',
		description='AI-powered data analysis and chart generation via REST.',
		version='1.0.0',
	)

	@app.post('/chat', response_model=ChatResponse)
	async def chat_endpoint(request: ChatRequest):
		"""Execute a natural-language query against the provided data."""
		from .smart_df import SmartDataFrame
		from .config import config as global_config

		if llm is not None:
			global_config.set({'llm': llm})

		if global_config.get('llm') is None:
			raise HTTPException(
				status_code=400,
				detail='No LLM configured. Set one via create_app(llm=...) or config.',
			)

		if request.data is None:
			raise HTTPException(status_code=400, detail='No data provided.')

		df = pd.DataFrame(request.data)
		sdf = SmartDataFrame(df)
		result = sdf.chat(
			request.query,
			chart_library=request.chart_library,
			agent='own',
			explain=request.explain,
		)

		is_chart = isinstance(result, str) and result.endswith(('.png', '.html', '.svg'))
		return ChatResponse(
			result=result,
			is_chart=is_chart,
			chart_path=result if is_chart else None,
		)

	@app.post('/chat/upload', response_model=ChatResponse)
	async def chat_upload_endpoint(
		query: str = Form(...),
		file: UploadFile = File(...),
		chart_library: str = Form('seaborn'),
		explain: bool = Form(False),
	):
		"""Upload a CSV/JSON file and execute a query against it."""
		from .smart_df import SmartDataFrame
		from .config import config as global_config

		if global_config.get('llm') is None:
			raise HTTPException(status_code=400, detail='No LLM configured.')

		content = await file.read()
		filename = file.filename or 'data.csv'

		if filename.endswith('.json'):
			df = pd.read_json(io.BytesIO(content))
		elif filename.endswith(('.xlsx', '.xls')):
			df = pd.read_excel(io.BytesIO(content))
		else:
			df = pd.read_csv(io.BytesIO(content))

		sdf = SmartDataFrame(df)
		result = sdf.chat(
			query,
			chart_library=chart_library,
			agent='own',
			explain=explain,
		)

		is_chart = isinstance(result, str) and result.endswith(('.png', '.html', '.svg'))
		return ChatResponse(
			result=result,
			is_chart=is_chart,
			chart_path=result if is_chart else None,
		)

	@app.get('/chart/{path:path}')
	async def get_chart(path: str):
		"""Serve a generated chart file."""
		if not os.path.isfile(path):
			raise HTTPException(status_code=404, detail='Chart not found.')
		return FileResponse(path)

	@app.post('/profile', response_model=ProfileResponse)
	async def profile_endpoint(data: List[Dict[str, Any]]):
		"""Generate a data profile report."""
		from .profiler import DataProfiler

		df = pd.DataFrame(data)
		report = DataProfiler.profile(df)
		return ProfileResponse(
			n_rows=report.n_rows,
			n_columns=report.n_columns,
			memory_usage_mb=report.memory_usage_mb,
			dtypes=report.dtypes,
			duplicates=report.duplicates,
			constant_columns=report.constant_columns,
			high_cardinality=report.high_cardinality,
			summary=report.summary,
		)

	@app.get('/health')
	async def health():
		"""Health check endpoint."""
		from .config import config as global_config
		return {
			'status': 'ok',
			'llm_configured': global_config.get('llm') is not None,
		}

	return app


# Default app instance for uvicorn
app = create_app()
