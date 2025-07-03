import shlex
import subprocess
from pathlib import Path

import modal

parent_dir = Path(__file__).parent
project_root = parent_dir.parent
streamlit_script_local_path = parent_dir / "modal_streamlit.py"
streamlit_script_remote_path = "/root/modal_streamlit.py"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "chromadb>=1.0.0",
        "langchain==0.3.25",
        "langchain-chroma>=0.2.4",
        "langchain-community>=0.3.25",
        "langchain-groq>=0.3.2",
        "langchain-huggingface==0.3.0",
        "langchain-openai>=0.3.22",
        "numpy>=2.0.0",
        "omegaconf>=2.3.0",
        "pandas>=2.3.0",
        "protobuf==5.29.5",
        "pydantic>=2.0.0",
        "pypdf>=5.6.0",
        "python-dotenv>=1.0.0",
        "pysqlite3-binary>=0.5.2",
        "rank-bm25>=0.2.2",
        "sentence-transformers>=4.1.0",
        "streamlit>=1.46.0",
        "scikit-learn>=1.7.0",
        )
    .env({  
        "DEPLOYMENT_MODE": "cloud",
        "IN_MEMORY": "true", 
        "DEBUG_MODE": "false",
        "PYTHONPATH": "/root"
    })
    .add_local_file(
        streamlit_script_local_path,
        streamlit_script_remote_path,
    )
    .add_local_dir(project_root / "frontend", "/root/frontend")
    .add_local_dir(project_root / "data/sample_pdfs", "/root/data/sample_pdfs")
    .add_local_dir(project_root / "backend", "/root/backend")
    .add_local_dir(project_root / "configs", "/root/configs")
    .add_local_dir(project_root / ".streamlit", "/root/.streamlit")
    .add_local_file(project_root / "frontend/static/image.jpeg", "/root/frontend/static/image.jpeg")
)

app = modal.App(name="two-stage-conrag", image=image)

if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "modal_streamlit.py not found! Place the script with your streamlit app in the same directory."
    )

@app.function(
        gpu="A10G:1",
        secrets=[modal.Secret.from_name("groq-secret")]
        )
@modal.concurrent(max_inputs=100)
@modal.web_server(8000)
def run():
    target = shlex.quote(streamlit_script_remote_path)
    cmd = f"""streamlit run {target} \
        --server.port 8000 \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false \
        --server.headless=true"""
    subprocess.Popen(cmd, shell=True)

