import os
import sys
import uuid
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory

sys.path.insert(0, '/app')

app = Flask(__name__)

# Store job status
jobs = {}

def run_generation(job_id, mode, prompt=None, image_path=None, use_sharp=False, return_mesh=False):
    """Run WorldGen generation in background thread."""
    try:
        jobs[job_id]['status'] = 'running'
        jobs[job_id]['message'] = 'Loading models...'

        from worldgen import WorldGen
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        jobs[job_id]['message'] = 'Initializing WorldGen...'
        worldgen = WorldGen(mode=mode, device=device)

        jobs[job_id]['message'] = 'Generating scene...'

        if mode == "t2s":
            result = worldgen.generate_world(prompt, use_sharp=use_sharp, return_mesh=return_mesh)
        else:
            result = worldgen.generate_world(image_path, use_sharp=use_sharp, return_mesh=return_mesh)

        # Save output
        output_filename = f"{job_id}.ply"
        output_path = f"/app/output/{output_filename}"
        result.save(output_path)

        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['message'] = 'Generation complete!'
        jobs[job_id]['output_file'] = output_filename

    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['message'] = str(e)

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
    app.run(host='0.0.0.0', port=5000, debug=False)
