import os
import sys
import uuid
import threading
import traceback
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory

sys.path.insert(0, '/app')

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Store job status
jobs = {}

def run_generation(job_id, mode, prompt=None, image_path=None, use_sharp=False, return_mesh=False):
    """Run WorldGen generation in background thread."""
    try:
        logger.info(f"[{job_id}] Starting generation - mode: {mode}, prompt: {prompt[:50] if prompt else image_path}")
        jobs[job_id]['status'] = 'running'
        jobs[job_id]['message'] = 'Loading models...'

        logger.info(f"[{job_id}] Importing WorldGen...")
        from worldgen import WorldGen
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[{job_id}] Using device: {device}")

        if device == "cuda":
            logger.info(f"[{job_id}] GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"[{job_id}] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        jobs[job_id]['message'] = f'Initializing WorldGen on {device}...'
        logger.info(f"[{job_id}] Initializing WorldGen...")
        worldgen = WorldGen(mode=mode, device=device)

        jobs[job_id]['message'] = 'Generating scene...'
        logger.info(f"[{job_id}] Starting scene generation...")

        if mode == "t2s":
            result = worldgen.generate_world(prompt, use_sharp=use_sharp, return_mesh=return_mesh)
        else:
            result = worldgen.generate_world(image_path, use_sharp=use_sharp, return_mesh=return_mesh)

        # Save output
        output_filename = f"{job_id}.ply"
        output_path = f"/app/output/{output_filename}"
        result.save(output_path)
        logger.info(f"[{job_id}] Saved output to {output_path}")

        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['message'] = 'Generation complete!'
        jobs[job_id]['output_file'] = output_filename
        logger.info(f"[{job_id}] Generation completed successfully")

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{job_id}] Generation failed: {error_msg}")
        logger.error(traceback.format_exc())
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

    logger.info(f"[{job_id}] Job created - {mode}: {prompt if mode == 't2s' else image_path}")

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
    logger.info("Starting WorldGen web server...")
    logger.info("Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("No GPU detected - generation will be very slow or fail")
    except Exception as e:
        logger.error(f"Error checking GPU: {e}")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
