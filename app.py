import uuid
import time
import threading
import shutil
from flask import make_response
from flask import Flask, render_template, request, jsonify, url_for
import os
import tempfile
from ingest import ingest
from rag import RAGService
from werkzeug.utils import secure_filename


TMP_DIR = tempfile.gettempdir()


def get_session_id():
    sid = request.cookies.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
    return sid

def get_vectordb_path(sid):
    return f"/tmp/vectordb_{sid}"

def cleanup_tmp_dbs(ttl_minutes=30):
    while True:
        now = time.time()
        for d in os.listdir(TMP_DIR):
            if d.startswith("vectordb_") or d.startswith("uploads_"):
                path = os.path.join("/tmp", d)
                if os.path.isdir(path):
                    if now - os.path.getmtime(path) > ttl_minutes * 60:
                        shutil.rmtree(path, ignore_errors=True)
        time.sleep(300)


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"


# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/")
def index():
    sid = get_session_id()
    resp = make_response(render_template("index.html"))
    resp.set_cookie("sid", sid, max_age=60 * 30)  # 30 min session
    return resp


@app.route("/chat_page")
def chat_page():
    return render_template("chat.html")

from langchain_core.documents import Document # Add this import at the top

@app.route("/ingest", methods=["POST"])
def ingest_data():
    sid = get_session_id()
    vectordb_path = get_vectordb_path(sid)

    files = request.files.getlist("files")
    url = request.form.get("url")
    yt_url = request.form.get("yt_url")
    raw_text_content = request.form.get("raw_text")

    upload_dir = f"/tmp/uploads_{sid}"
    os.makedirs(upload_dir, exist_ok=True)

    saved_pdfs = []
    saved_texts = []

    for file in files:
        if file.filename:
            fn = secure_filename(file.filename)
            path = os.path.join(upload_dir, fn)
            file.save(path)
            if fn.lower().endswith(".pdf"):
                saved_pdfs.append(path)
            else:
                saved_texts.append(path)

    if raw_text_content and raw_text_content.strip():
        raw_text_path = os.path.join(upload_dir, "pasted.txt")
        with open(raw_text_path, "w", encoding="utf-8") as f:
            f.write(raw_text_content)
        saved_texts.append(raw_text_path)

    ingest(
        persist_directory=vectordb_path,
        pdf_files=saved_pdfs or None,
        text_files=saved_texts or None,
        urls=[url] if url else None,
        youtube_links=[yt_url] if yt_url else None,
        reset=True   # ✅ SAFE now
    )

    return jsonify({"status": "success", "redirect": url_for("chat_page")})


@app.route("/ask", methods=["POST"])
def ask():
    sid = get_session_id()
    vectordb_path = get_vectordb_path(sid)

    if not os.path.exists(vectordb_path):
        return jsonify({"error": "Session expired. Please upload documents again."}), 400

    rag = RAGService(persist_directory=vectordb_path)
    query = request.json.get("query")

    result = rag.ask(query, session_id=sid)

    return jsonify({
        "status": "success",
        "answer": result["answer"],
        "sources": result["sources"]
    })


# cleanup thread
threading.Thread(target=cleanup_tmp_dbs, daemon=True).start()


if __name__ == "__main__":
    app.run(debug=True)
