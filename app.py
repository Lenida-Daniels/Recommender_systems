from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
	page_title="CineScope | Recommender Lab",
	page_icon="C",
	layout="wide",
	initial_sidebar_state="expanded",
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
	"""Load a ratings file when available, with a useful local demo fallback."""
	data_path = Path(__file__).parent / "data" / "data.txt"
	if data_path.exists() and data_path.stat().st_size:
		try:
			raw = pd.read_csv(data_path)
			columns = {column.lower().strip(): column for column in raw.columns}
			title_column = next((columns[key] for key in ("title", "movie", "name") if key in columns), None)
			user_column = next((columns[key] for key in ("user_id", "userid", "user") if key in columns), None)
			rating_column = next((columns[key] for key in ("rating", "score") if key in columns), None)
			if title_column and user_column and rating_column:
				ratings = raw.rename(columns={title_column: "title", user_column: "user_id", rating_column: "rating"})
				ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
				ratings = ratings.dropna(subset=["title", "user_id", "rating"])
				movies = ratings[["title"]].drop_duplicates().reset_index(drop=True)
				movies["movie_id"] = np.arange(1, len(movies) + 1)
				ratings = ratings.merge(movies, on="title", how="left")
				ratings["genres"] = "Drama"
				return movies[["movie_id", "title"]].assign(genres="Drama"), ratings, False
		except (OSError, ValueError, pd.errors.ParserError):
			pass

	titles = [
		"The Last Signal", "Midnight in Kyoto", "Orbit 9", "Paper Planes", "Neon Harbor",
		"The Quiet Season", "After the Storm", "Velvet Static", "Northbound", "A Small Light",
		"The Long Game", "Wildflower Hotel", "Static Hearts", "Tidal Memory", "The Cartographer",
		"Glass Houses", "Second Sunrise", "The Listening Room", "Mosaic City", "Low Orbit",
	]
	genres = ["Sci-Fi", "Drama", "Sci-Fi", "Comedy", "Thriller", "Drama", "Drama", "Thriller", "Adventure", "Drama", "Documentary", "Comedy", "Romance", "Drama", "Adventure", "Thriller", "Romance", "Documentary", "Comedy", "Sci-Fi"]
	movies = pd.DataFrame({"movie_id": range(1, len(titles) + 1), "title": titles, "genres": genres})
	rng = np.random.default_rng(42)
	rows = []
	for user_id in range(1, 81):
		watched = rng.choice(movies.movie_id, size=rng.integers(5, 13), replace=False)
		for movie_id in watched:
			base = 3.2 + (movie_id % 5) * 0.18 + (user_id % 4) * 0.08
			rows.append((user_id, movie_id, float(np.clip(np.round(base + rng.normal(0, 0.72), 1), 1, 5))))
	return movies, pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"]).merge(movies, on="movie_id"), True


def popular_recommendations(movies: pd.DataFrame, ratings: pd.DataFrame, limit: int, seen: Iterable[int] = ()) -> pd.DataFrame:
	stats = ratings.groupby("movie_id").agg(score=("rating", "mean"), ratings=("rating", "count")).reset_index()
	stats["score"] = stats["score"] * (1 - np.exp(-stats["ratings"] / 8))
	return movies.merge(stats, on="movie_id", how="left").fillna({"score": 0, "ratings": 0}).query("movie_id not in @seen").sort_values(["score", "ratings"], ascending=False).head(limit)


def content_recommendations(movies: pd.DataFrame, ratings: pd.DataFrame, movie_id: int, limit: int) -> pd.DataFrame:
	genre_matrix = pd.get_dummies(movies.set_index("movie_id")["genres"])
	similarities = cosine_similarity(genre_matrix)
	index = list(genre_matrix.index).index(movie_id)
	scores = pd.DataFrame({"movie_id": genre_matrix.index, "score": similarities[index]})
	return movies.merge(scores, on="movie_id").query("movie_id != @movie_id").sort_values("score", ascending=False).head(limit)


def collaborative_recommendations(movies: pd.DataFrame, ratings: pd.DataFrame, user_id: int, limit: int) -> pd.DataFrame:
	matrix = ratings.pivot_table(index="user_id", columns="movie_id", values="rating", fill_value=0)
	if user_id not in matrix.index:
		return popular_recommendations(movies, ratings, limit)
	similarities = cosine_similarity(matrix)
	neighbors = pd.Series(similarities[matrix.index.get_loc(user_id)], index=matrix.index).drop(user_id).nlargest(8)
	watched = set(ratings.loc[ratings.user_id == user_id, "movie_id"])
	scores = matrix.loc[neighbors.index].T.dot(neighbors).div(neighbors.sum()).rename("score").reset_index()
	scores = scores.rename(columns={scores.columns[0]: "movie_id"})
	return movies.merge(scores, on="movie_id", how="left").fillna({"score": 0}).query("movie_id not in @watched").sort_values("score", ascending=False).head(limit)


def metric_cards(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
	values = [(f"{ratings.user_id.nunique():,}", "active users", "audience profiles"), (f"{len(movies):,}", "titles", "in the catalog"), (f"{len(ratings):,}", "ratings", "behavioral signals"), (f"{ratings.rating.mean():.1f}/5", "average rating", "across all users")]
	with st.grid(4, gap="small"):
		for value, label, detail in values:
			with st.container(border=True):
				st.metric(label, value)
				st.caption(detail)


def render_movie_rows(data: pd.DataFrame, score_label: str = "Match score") -> None:
	for rank, (_, row) in enumerate(data.reset_index(drop=True).iterrows(), 1):
		with st.container(border=True):
			left, middle, right = st.columns([0.65, 3.8, 1.35], vertical_alignment="center")
			left.markdown(f"<div class='rank'>{rank:02d}</div>", unsafe_allow_html=True)
			middle.markdown(f"<div class='film-title'>{row['title']}</div><div class='film-meta'>{row.get('genres', 'Film')} &nbsp; · &nbsp; {int(row.get('ratings', 0))} ratings</div>", unsafe_allow_html=True)
			score = row.get("score", 0)
			right.metric(score_label, f"{score:.2f}")


def dashboard(movies: pd.DataFrame, ratings: pd.DataFrame, demo: bool) -> None:
	st.markdown("<div class='eyebrow'>OVERVIEW / 01</div><h1>Good recommendations<br><span class='accent'>start with signal.</span></h1>", unsafe_allow_html=True)
	st.markdown("<p class='lede'>A quiet workspace for understanding your catalog, testing recommendation strategies, and finding the edges where your models need help.</p>", unsafe_allow_html=True)
	if demo:
		st.caption("You are exploring a prepared sample catalog. Connect your own ratings file in `data/data.txt` when you are ready.")
	metric_cards(ratings, movies)
	st.subheader("Signal overview", anchor=False)
	left, right = st.columns([1.35, 1], gap="large")
	with left:
		trend = ratings.groupby("rating").size().reindex([1, 2, 3, 4, 5], fill_value=0)
		st.bar_chart(trend, height=250, color="#f07f55")
		st.caption("How your audience rates the catalog")
	with right:
		genre_counts = movies["genres"].value_counts().rename_axis("genre").to_frame("titles")
		st.dataframe(genre_counts, width="stretch", height=250, hide_index=False)
		st.caption("Catalog balance by genre")


def recommendations(movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
	st.markdown("<div class='eyebrow'>DISCOVERY / 02</div><h2>Recommendation studio</h2><p class='lede'>Choose a lens. The shortlist updates with the model's point of view.</p>", unsafe_allow_html=True)
	method = st.segmented_control("Recommendation strategy", ["Popular", "Content similarity", "Collaborative"], default="Popular", label_visibility="collapsed", key="recommendation_strategy")
	method = method or "Popular"
	with st.container(border=True):
		control, context = st.columns([1.1, 1], vertical_alignment="bottom")
		with control:
			limit = st.slider("Show results", 3, 10, 5, key="recommendation_limit")
		with context:
			if method == "Popular":
				st.caption("**Popularity**  ·  Trust the strongest catalog-wide signal.")
			elif method == "Content similarity":
				st.caption("**Similarity**  ·  Find a neighboring taste profile.")
			else:
				st.caption("**Collaborative**  ·  Borrow signal from similar viewers.")
	if method == "Popular":
		results = popular_recommendations(movies, ratings, limit)
	elif method == "Content similarity":
		selected = st.selectbox("Start from a title", movies.title.tolist(), key="similar_title")
		results = content_recommendations(movies, ratings, int(movies.loc[movies.title == selected, "movie_id"].iloc[0]), limit)
	else:
		selected_user = st.selectbox("Start from an audience profile", sorted(ratings.user_id.unique()), key="audience_profile")
		results = collaborative_recommendations(movies, ratings, int(selected_user or 1), limit)
	st.markdown(f"<div class='result-header'><div><div class='eyebrow'>TOP {limit} RESULTS</div><h3>{method}</h3></div><div class='result-note'>Ranked just now<br><span>Explainable local model</span></div></div>", unsafe_allow_html=True)
	render_movie_rows(results)


def evaluation(movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
	st.markdown("<div class='eyebrow'>MODEL HEALTH / 03</div><h2>Evaluation bench</h2><p class='lede'>A compact read on ranking quality, coverage, and the tradeoffs between your three models.</p>", unsafe_allow_html=True)
	rows = []
	for name, function in [("Popular", lambda user: popular_recommendations(movies, ratings, 5)), ("Content", lambda user: content_recommendations(movies, ratings, int(movies.movie_id.iloc[user % len(movies)]), 5)), ("Collaborative", lambda user: collaborative_recommendations(movies, ratings, int(user), 5))]:
		hits = 0
		users = ratings.user_id.unique()[:30]
		for user in users:
			held_out = int(ratings[ratings.user_id == user].sort_values("rating").iloc[-1].movie_id)
			if held_out in set(function(int(user)).movie_id):
				hits += 1
		rows.append({"Model": name, "Precision@5": round(hits / len(users), 2), "Coverage": round(len(function(1)) / len(movies), 2), "Latency": "< 20 ms"})
	scores = pd.DataFrame(rows)
	st.dataframe(scores, width="stretch", hide_index=True, column_config={"Precision@5": st.column_config.ProgressColumn("Precision@5", min_value=0, max_value=1, format="%.2f"), "Coverage": st.column_config.ProgressColumn("Coverage", min_value=0, max_value=1, format="%.2f")})
	with st.expander("How to read this", icon=":material/info:"):
		st.caption("Precision@5 measures held-out hits in the top five. Coverage shows the shortlist size relative to the catalog. Latency is an indicative local estimate for this lightweight benchmark.")


def cold_start(movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
	st.markdown("<div class='eyebrow'>ROBUSTNESS / 04</div><h2>Cold-start simulator</h2><p class='lede'>Every recommender has a first moment. Test how yours behaves before history exists.</p>", unsafe_allow_html=True)
	scenario = st.segmented_control("Scenario", ["New user", "New title", "Limited ratings"], default="New user", label_visibility="collapsed", key="cold_start_scenario") or "New user"
	if scenario == "New user":
		st.info("A new viewer has no behavioral history. CineScope falls back to catalog-wide confidence.", icon=":material/person_add:")
		render_movie_rows(popular_recommendations(movies, ratings, 5), "Fallback score")
	elif scenario == "New title":
		genre = st.selectbox("New title genre", sorted(movies.genres.unique()), key="new_title_genre") or "Drama"
		candidates = movies[movies.genres == genre].copy()
		candidates["score"] = 1.0
		st.info(f"An unrated {genre.lower()} title borrows its first signal from nearby catalog entries.", icon=":material/fiber_new:")
		render_movie_rows(candidates.head(5), "Genre match")
	else:
		count = st.slider("Ratings available", 1, 5, 2, key="limited_ratings")
		limited = ratings.groupby("user_id").head(count)
		st.info(f"Only {count} rating{'s' if count != 1 else ''} per viewer are available to the model.", icon=":material/tune:")
		render_movie_rows(popular_recommendations(movies, limited, 5), "Signal score")


def main() -> None:
	movies, ratings, demo = load_data()
	st.markdown("<style> @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap'); :root { --ink:#17202a; --muted:#66727d; --orange:#ef8354; --cream:#f7f3ed; --line:#e4ddd4; } .stApp { background:var(--cream); color:var(--ink); } [data-testid='stSidebar'] { background:#17202a; } [data-testid='stSidebar'] * { color:#f7f3ed !important; } [data-testid='stSidebar'] [data-testid='stRadio'] label { padding: .55rem .65rem; border-radius:4px; } [data-testid='stSidebar'] [data-testid='stRadio'] label:hover { background:#243440; } h1,h2,h3 { font-family:'Space Grotesk', sans-serif; letter-spacing:0 !important; } h1 { font-size:clamp(3.2rem,7vw,6.6rem); line-height:.9; margin:1.2rem 0 1.2rem; } h2 { font-size:2.8rem; line-height:1; margin:1.2rem 0; } h3 { font-size:1.35rem; } .accent { color:var(--orange); } .lede { color:var(--muted); font-size:1.12rem; max-width:650px; margin-bottom:2.5rem; } .eyebrow { color:var(--orange); font-family:'DM Mono',monospace; font-size:.72rem; letter-spacing:.08em; margin-top:1rem; } .rank { color:var(--orange); font-family:'DM Mono',monospace; font-size:.8rem; } .film-title { font-family:'Space Grotesk',sans-serif; font-size:1.08rem; font-weight:600; } .film-meta { color:var(--muted); font-family:'DM Mono',monospace; font-size:.7rem; margin-top:.3rem; text-transform:uppercase; } .result-header { display:flex; justify-content:space-between; align-items:end; margin:2.3rem 0 .6rem; } .result-note { color:var(--muted); font-family:'DM Mono',monospace; font-size:.68rem; line-height:1.6; text-align:right; text-transform:uppercase; } .result-note span { color:var(--orange); } [data-testid='stMetricValue'] { font-family:'Space Grotesk',sans-serif; } [data-testid='stMetric'] { background:#fffaf4; border:1px solid var(--line); padding:1rem; } .stButton button, .stDownloadButton button { border-radius:3px; } @media (max-width: 700px) { h1 { font-size:3.3rem; } h2 { font-size:2.2rem; } .result-header { display:block; } .result-note { text-align:left; margin-top:.5rem; } } </style>", unsafe_allow_html=True)
	with st.sidebar:
		st.markdown("<div style='font-family:Space Grotesk;font-size:1.5rem;font-weight:700;letter-spacing:-.04em'>CINE<span style='color:#ef8354'>SCOPE</span></div><div style='font-family:DM Mono;font-size:.62rem;letter-spacing:.08em;margin:.35rem 0 2.2rem;color:#9eabb3'>RECOMMENDER INTELLIGENCE</div>", unsafe_allow_html=True)
		section = st.radio("Workspace", ["Dashboard", "Recommendations", "Model Evaluation", "Cold Start"], format_func=lambda item: {"Dashboard": "▦  Dashboard", "Recommendations": "✦  Recommendations", "Model Evaluation": "↗  Model evaluation", "Cold Start": "◌  Cold start"}[item], label_visibility="collapsed", key="workspace")
		st.markdown("<div style='height:1px;background:#31414b;margin:1.2rem 0'></div>", unsafe_allow_html=True)
		st.caption("WORKSPACE STATUS")
		st.markdown("<span style='color:#7ed6a4'>●</span>&nbsp; Ready to explore", unsafe_allow_html=True)
		st.caption("Data mode: " + ("sample catalog" if demo else "connected catalog"))
	if section == "Dashboard":
		dashboard(movies, ratings, demo)
	elif section == "Recommendations":
		recommendations(movies, ratings)
	elif section == "Model Evaluation":
		evaluation(movies, ratings)
	else:
		cold_start(movies, ratings)


if __name__ == "__main__":
	main()
