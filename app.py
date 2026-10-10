import os
from flask import Flask, request, jsonify
import librosa
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CHORD_LABELS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Función para agregar cabeceras CORS a todas las respuestas
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

@app.route('/')
def index():
    return "Servidor de Acordes Activo 🎸"

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_audio():
    # Responder a las peticiones pre-vuelo (OPTIONS) de CORS
    if request.method == 'OPTIONS':
        return '', 200

    if 'file' not in request.files:
        return jsonify({'error': 'No se subió ningún archivo'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Archivo sin nombre'}), 400
        
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            y, sr = librosa.load(filepath, sr=22050, duration=60.0)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            hop_length = 512
            frames_per_second = sr / hop_length
            
            chords_detected = []
            block_size = int(frames_per_second)
            
            for i in range(0, chroma.shape[1], block_size):
                chunk = chroma[:, i:i+block_size]
                if chunk.shape[1] == 0:
                    break
                mean_chroma = np.mean(chunk, axis=1)
                best_chord_idx = np.argmax(mean_chroma)
                
                time_sec = round(i / frames_per_second, 1)
                chord_name = CHORD_LABELS[best_chord_idx]
                
                if not chords_detected or chords_detected[-1]['chord'] != chord_name:
                    chords_detected.append({
                        'time': time_sec,
                        'chord': chord_name
                    })
            
            os.remove(filepath)
            return jsonify({'success': True, 'chords': chords_detected})
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
