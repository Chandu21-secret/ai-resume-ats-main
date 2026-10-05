RENDER FREE (512 MB) OPTIMIZATION PATCH

Replace these files in your existing ai-resume-ats-main project, preserving the same paths:

backend/main.py
backend/core/config.py
backend/services/ats_scorer.py
backend/services/jd_matcher.py
backend/services/resume_analyzer.py
backend/services/lightweight_embedder.py
requirements.txt

What changed:
- Uses spaCy en_core_web_sm instead of en_core_web_md.
- Removes sentence-transformers/PyTorch from the runtime.
- Adds a deterministic lightweight hashed text embedder with the same encode() interface.
- Keeps the existing API routes and scoring pipeline structure.

Render Build Command:
pip install -r requirements.txt && python -m spacy download en_core_web_sm

Render Start Command:
uvicorn backend.main:app --host 0.0.0.0 --port $PORT

After replacing files in the local project:
git add .
git commit -m "Optimize backend for Render free tier"
git push

Note: semantic_similarity is now lightweight lexical similarity rather than the original SentenceTransformer embedding similarity, so semantic/JD scores can differ somewhat. Other application features and API structure are preserved.
