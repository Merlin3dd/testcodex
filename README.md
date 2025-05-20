# Elevation Tile Server

This Flask application serves satellite tiles from the `/maps` directory and downloads free elevation tiles on demand. It also provides a simple API to calculate the visible horizon.

## Usage (English)

1. Place `leaflet.js` and `leaflet.css` in the `static` folder. They can be obtained from [https://leafletjs.com](https://leafletjs.com).
2. Run the application:

```bash
python server.py
```

3. Open `http://localhost:5004/` in a browser. Click two points on the map to select an area and download elevation tiles. The server stores them under `/elevation`.

### Available routes

- `/tiles/<z>/<x>/<y>.jpg` – satellite or hybrid tiles you provide in `/maps`.
- `/elevation/<z>/<x>/<y>.png` – elevation tiles. Missing tiles are downloaded automatically from `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/`.
- `/download?z=<z>&lat1=<lat>&lon1=<lon>&lat2=<lat>&lon2=<lon>` – fetch a rectangle of elevation tiles.
- `/horizon?lat=<lat>&lon=<lon>&alt=<meters>` – returns a GeoJSON polygon of the horizon at the given altitude.

The application creates a placeholder tile if the network is unreachable.

## Инструкция на русском

1. Скопируйте файлы `leaflet.js` и `leaflet.css` в папку `static` (их можно взять на сайте [https://leafletjs.com](https://leafletjs.com)).
2. Запустите сервер командой:

```bash
python server.py
```

3. Откройте в браузере `http://localhost:5004/`. Кликните два раза по карте, чтобы выбрать прямоугольник. После этого сервер скачает тайлы высот в каталог `/elevation`.

Маршруты API те же, что описаны выше.

The server listens on port `5004` by default.
