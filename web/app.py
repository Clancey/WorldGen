import os
import sys
import uuid
import threading
import traceback
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory

# Configure logging FIRST - before anything else
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger('worldgen-web')
logger.setLevel(logging.DEBUG)

# Also capture all other loggers
for name in ['transformers', 'diffusers', 'torch', 'urllib3', 'huggingface_hub']:
    logging.getLogger(name).setLevel(logging.INFO)

print("=" * 60, flush=True)
print("WORLDGEN WEB SERVER STARTING", flush=True)
print("=" * 60, flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Working dir: {os.getcwd()}", flush=True)
print(f"HF_HOME: {os.environ.get('HF_HOME', 'not set')}", flush=True)
print(f"HF_TOKEN set: {bool(os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'))}", flush=True)
print("=" * 60, flush=True)

sys.path.insert(0, '/app')

app = Flask(__name__)

# Store job status
jobs = {}

def run_generation(job_id, mode, prompt=None, image_path=None, use_sharp=False, return_mesh=False):
    """Run WorldGen generation in background thread."""
    print(f"\n{'='*60}", flush=True)
    print(f"[{job_id}] GENERATION STARTED", flush=True)
    print(f"[{job_id}] Mode: {mode}", flush=True)
    print(f"[{job_id}] Prompt: {prompt}", flush=True)
    print(f"[{job_id}] Use sharp: {use_sharp}", flush=True)
    print(f"[{job_id}] Return mesh: {return_mesh}", flush=True)
    print(f"{'='*60}\n", flush=True)

    try:
        jobs[job_id]['status'] = 'running'
        jobs[job_id]['message'] = 'Importing torch...'
        print(f"[{job_id}] Importing torch...", flush=True)

        import torch
        print(f"[{job_id}] Torch version: {torch.__version__}", flush=True)
        print(f"[{job_id}] CUDA available: {torch.cuda.is_available()}", flush=True)

        if torch.cuda.is_available():
            print(f"[{job_id}] CUDA version: {torch.version.cuda}", flush=True)
            print(f"[{job_id}] GPU: {torch.cuda.get_device_name(0)}", flush=True)
            print(f"[{job_id}] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB", flush=True)
            device = "cuda"
        else:
            print(f"[{job_id}] WARNING: No CUDA - using CPU (will be very slow)", flush=True)
            device = "cpu"

        jobs[job_id]['message'] = 'Importing WorldGen...'
        print(f"[{job_id}] Importing WorldGen...", flush=True)

        from worldgen import WorldGen
        print(f"[{job_id}] WorldGen imported successfully", flush=True)

        jobs[job_id]['message'] = f'Initializing WorldGen on {device}...'
        print(f"[{job_id}] Creating WorldGen instance (this will download models)...", flush=True)

        worldgen = WorldGen(mode=mode, device=device)
        print(f"[{job_id}] WorldGen initialized", flush=True)

        jobs[job_id]['message'] = 'Generating scene...'
        print(f"[{job_id}] Starting generation...", flush=True)

        if mode == "t2s":
            result = worldgen.generate_world(prompt, use_sharp=use_sharp, return_mesh=return_mesh)
        else:
            result = worldgen.generate_world(image_path, use_sharp=use_sharp, return_mesh=return_mesh)

        print(f"[{job_id}] Generation complete, saving...", flush=True)

        # Save output
        output_filename = f"{job_id}.ply"
        output_path = f"/app/output/{output_filename}"
        result.save(output_path)
        print(f"[{job_id}] Saved to {output_path}", flush=True)

        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['message'] = 'Generation complete!'
        jobs[job_id]['output_file'] = output_filename
        print(f"[{job_id}] SUCCESS!", flush=True)

    except Exception as e:
        error_msg = str(e)
        print(f"\n[{job_id}] ERROR: {error_msg}", flush=True)
        print(f"[{job_id}] Full traceback:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['message'] = error_msg

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    mode = data.get('mode', 't2s')
    prompt = data.get('prompt', '')
    image_path = data.get('image_path', '')
    use_sharp = data.get('use_sharp', False)
    return_mesh = data.get('return_mesh', False)

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        'id': job_id,
        'status': 'queued',
        'message': 'Job queued',
        'created': datetime.now().isoformat(),
        'mode': mode,
        'prompt': prompt if mode == 't2s' else image_path,
        'output_file': None
    }

    print(f"\n[{job_id}] New job received: {mode} - {prompt if mode == 't2s' else image_path}", flush=True)

    thread = threading.Thread(
        target=run_generation,
        args=(job_id, mode, prompt, image_path, use_sharp, return_mesh)
    )
    thread.start()

    return jsonify({'job_id': job_id})

@app.route('/status/<job_id>')
def status(job_id):
    if job_id in jobs:
        return jsonify(jobs[job_id])
    return jsonify({'error': 'Job not found'}), 404

@app.route('/jobs')
def list_jobs():
    return jsonify(list(jobs.values()))

@app.route('/gpu')
def gpu_status():
    """Check GPU status and availability."""
    try:
        import torch
        if torch.cuda.is_available():
            return jsonify({
                'available': True,
                'device': torch.cuda.get_device_name(0),
                'vram_total_gb': round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
                'vram_used_gb': round(torch.cuda.memory_allocated(0) / 1024**3, 2),
                'vram_cached_gb': round(torch.cuda.memory_reserved(0) / 1024**3, 2),
            })
        else:
            return jsonify({'available': False, 'error': 'CUDA not available'})
    except Exception as e:
        return jsonify({'available': False, 'error': str(e)})

@app.route('/env')
def env_status():
    """Check environment variables and paths."""
    return jsonify({
        'hf_home': os.environ.get('HF_HOME', 'not set'),
        'hf_token_set': bool(os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')),
        'transformers_cache': os.environ.get('TRANSFORMERS_CACHE', 'not set'),
        'cwd': os.getcwd(),
        'app_files': os.listdir('/app') if os.path.exists('/app') else [],
        'output_files': os.listdir('/app/output') if os.path.exists('/app/output') else [],
        'hf_cache_exists': os.path.exists(os.environ.get('HF_HOME', '/root/.cache/huggingface')),
        'hf_cache_contents': os.listdir(os.environ.get('HF_HOME', '/root/.cache/huggingface')) if os.path.exists(os.environ.get('HF_HOME', '/root/.cache/huggingface')) else [],
    })

@app.route('/test')
def test_imports():
    """Test if all imports work."""
    results = {}

    try:
        import torch
        results['torch'] = f"OK - {torch.__version__}"
        results['cuda'] = f"{'Available' if torch.cuda.is_available() else 'NOT available'}"
    except Exception as e:
        results['torch'] = f"FAILED: {e}"

    try:
        import diffusers
        results['diffusers'] = f"OK - {diffusers.__version__}"
    except Exception as e:
        results['diffusers'] = f"FAILED: {e}"

    try:
        import transformers
        results['transformers'] = f"OK - {transformers.__version__}"
    except Exception as e:
        results['transformers'] = f"FAILED: {e}"

    try:
        from worldgen import WorldGen
        results['worldgen'] = "OK"
    except Exception as e:
        results['worldgen'] = f"FAILED: {e}"

    return jsonify(results)

@app.route('/output/<filename>')
def download_output(filename):
    return send_from_directory('/app/output', filename)

@app.route('/outputs')
def list_outputs():
    files = []
    output_dir = '/app/output'
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.endswith(('.ply', '.glb')):
                filepath = os.path.join(output_dir, f)
                files.append({
                    'name': f,
                    'size': os.path.getsize(filepath),
                    'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                })
    return jsonify(files)

if __name__ == '__main__':
    print("\n" + "=" * 60, flush=True)
    print("STARTUP CHECKS", flush=True)
    print("=" * 60, flush=True)

    # Check GPU
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}", flush=True)
        if torch.cuda.is_available():
            print(f"CUDA available: YES", flush=True)
            print(f"CUDA version: {torch.version.cuda}", flush=True)
            print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            print("CUDA available: NO - This will not work properly!", flush=True)
    except Exception as e:
        print(f"PyTorch import failed: {e}", flush=True)
        traceback.print_exc()

    # Check HF token
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    print(f"HF Token: {'SET' if hf_token else 'NOT SET - Models may fail to download!'}", flush=True)

    # Check paths
    print(f"HF cache dir: {os.environ.get('HF_HOME', '/root/.cache/huggingface')}", flush=True)

    # Test imports one by one to find which one crashes
    print("=" * 60, flush=True)
    print("Testing imports step by step...", flush=True)
    sys.stdout.flush()

    imports_to_test = [
        ("numpy", "import numpy"),
        ("PIL", "from PIL import Image"),
        ("torch", "import torch"),
        ("torchvision", "import torchvision"),
        ("einops", "import einops"),
        ("transformers", "import transformers"),
        ("diffusers", "import diffusers"),
        ("open3d", "import open3d"),
        ("trimesh", "import trimesh"),
        ("skimage", "import skimage"),
        ("py360convert", "import py360convert"),
        ("peft", "import peft"),
        ("worldgen", "from worldgen import WorldGen"),
    ]

    for name, import_stmt in imports_to_test:
        print(f"  Testing {name}...", end=" ", flush=True)
        try:
            exec(import_stmt)
            print("OK", flush=True)
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            traceback.print_exc()
        sys.stdout.flush()

    print("=" * 60, flush=True)
    print("Starting Flask server on port 5000...", flush=True)
    print("=" * 60 + "\n", flush=True)
    sys.stdout.flush()

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
