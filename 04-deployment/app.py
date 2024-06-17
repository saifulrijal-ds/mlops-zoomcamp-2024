from flask import Flask, request, jsonify
import subprocess
import json

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_script():
    try:
        data = request.get_json()
        taxi_type = data['taxi_type']
        year = data['year']
        month = data['month']

        result = subprocess.run(
            ['python', 'score.py', taxi_type, str(year), str(month)], 
            capture_output=True, 
            text=True
        )

        output_lines = result.stdout.strip().split('\n')

        if result.returncode == 0:
            mean = float(output_lines[0].split(': ')[1])
            std = float(output_lines[1].split(': ')[1])

            return jsonify({
                'mean_prediction': mean,
                'std_prediction': std,
                'stdout': result.stdout,
                'stderr': result.stderr
            })
        else:
             return jsonify({
                'error': 'Error running score.py',
                'stdout': result.stdout,
                'stderr': result.stderr
            }), 500
    except Exception as e:
        return jsonify({'error': str(e), 'stdout': result.stdout if 'result' in locals() else '', 'stderr': result.stderr if 'result' in locals() else ''}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
