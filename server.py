from flask import Flask, send_file, abort, request, jsonify
import os
import urllib.request
import base64
import math
from math import sqrt, sin, cos, atan2, radians, degrees, asin

ELEVATION_SOURCE = 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'
ELEVATION_DIR = 'elevation'
PLACEHOLDER_PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVQYV2NgYAAAAAMAASsJTYQAAAAASUVORK5CYII=')

app = Flask(__name__)

# Utility to download a single elevation tile or create a placeholder
def fetch_tile(z, x, y):
    path_dir = os.path.join(ELEVATION_DIR, str(z), str(x))
    os.makedirs(path_dir, exist_ok=True)
    dest = os.path.join(path_dir, f'{y}.png')
    if os.path.exists(dest):
        return dest
    url = ELEVATION_SOURCE.format(z=z, x=x, y=y)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception:
        with open(dest, 'wb') as f:
            f.write(PLACEHOLDER_PNG)
    return dest

# Prepare a sample tile at startup so the server works offline
fetch_tile(0, 0, 0)

# Serve satellite or hybrid map tiles (provided separately)
@app.route('/tiles/<int:z>/<int:x>/<int:y>.jpg')
def get_tile(z, x, y):
    tile_path = f'/maps/{z}/{x}/{y}.jpg'
    if os.path.exists(tile_path):
        return send_file(tile_path, mimetype='image/jpg')
    abort(404)

# Serve elevation tiles
@app.route('/elevation/<int:z>/<int:x>/<int:y>.png')
def get_elevation_tile(z, x, y):
    tile_path = os.path.join(ELEVATION_DIR, str(z), str(x), f'{y}.png')
    if not os.path.exists(tile_path):
        fetch_tile(z, x, y)
    if os.path.exists(tile_path):
        return send_file(tile_path, mimetype='image/png')
    abort(404)

# Download a range of elevation tiles defined by two lat/lon corners and zoom
@app.route('/download')
def download():
    try:
        z = int(request.args['z'])
        lat1 = float(request.args['lat1'])
        lon1 = float(request.args['lon1'])
        lat2 = float(request.args['lat2'])
        lon2 = float(request.args['lon2'])
    except (KeyError, ValueError):
        abort(400)

    def latlon_to_tile(lat, lon, zoom):
        n = 2 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1 / cos(math.radians(lat))) / math.pi) / 2.0 * n)
        return x, y

    x1, y1 = latlon_to_tile(lat1, lon1, z)
    x2, y2 = latlon_to_tile(lat2, lon2, z)
    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])

    count = 0
    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            fetch_tile(z, x, y)
            count += 1
    return jsonify({'downloaded': count})

# Calculate visible horizon from a given point and altitude, return GeoJSON
@app.route('/horizon')
def horizon():
    try:
        lat = float(request.args['lat'])
        lon = float(request.args['lon'])
        alt = float(request.args.get('alt', 0))
    except (KeyError, ValueError):
        abort(400)

    R = 6371000.0  # Earth radius in meters
    distance = sqrt(alt**2 + 2 * R * alt)
    features = []
    for az in range(0, 360, 5):
        bearing = radians(az)
        d = distance / R
        lat1 = radians(lat)
        lon1 = radians(lon)
        lat2 = asin(sin(lat1) * cos(d) + cos(lat1) * sin(d) * cos(bearing))
        lon2 = lon1 + atan2(sin(bearing) * sin(d) * cos(lat1),
                            cos(d) - sin(lat1) * sin(lat2))
        features.append([degrees(lat2), degrees(lon2)])

    geojson = {
        'type': 'FeatureCollection',
        'features': [{
            'type': 'Feature',
            'properties': {'distance_m': distance},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [features]
            }
        }]
    }
    return jsonify(geojson)

@app.route('/')
def index():
    return send_file('static/index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004)
