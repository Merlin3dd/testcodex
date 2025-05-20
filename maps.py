from flask import Flask, send_file, abort
import os

app = Flask(__name__)


@app.route('/tiles/<int:z>/<int:x>/<int:y>.jpg')
def get_tile(z, x, y):
    tile_path = f'/maps/{z}/{x}/{y}.jpg'
    if os.path.exists(tile_path):
        return send_file(tile_path, mimetype='image/jpg')
    else:
        abort(404)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004)
