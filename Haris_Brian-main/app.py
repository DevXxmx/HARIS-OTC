from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
from flask import Flask, jsonify, send_from_directory, Response
import os


app = Flask(__name__, static_folder='static', static_url_path='/static')

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///photos.db'


db = SQLAlchemy()
db.init_app(app)

# Allow all origins (frontend on :5173 talking to backend on :5000)
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/static/*": {"origins": "*"}})


class Photo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    filepath = db.Column(db.String(255), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "filepath": self.filepath,
            "upload_time": self.upload_time.strftime("%Y-%m-%dT%H:%M:%S")
        }


@app.route("/api/alerts")
def get_alerts():
    photos = Photo.query.order_by(Photo.upload_time.desc()).all()
    valid_photos = [p for p in photos if os.path.exists(p.filepath)]
    return jsonify([p.to_dict() for p in valid_photos])


@app.route("/api/alerts/latest")
def get_latest_alert():
    photos = Photo.query.order_by(Photo.upload_time.desc()).all()
    for p in photos:
        if os.path.exists(p.filepath):
            return jsonify(p.to_dict())
    return jsonify(None)


@app.route("/api/alerts/<int:alert_id>")
def get_alert(alert_id):
    photo = Photo.query.get_or_404(alert_id)
    return jsonify(photo.to_dict())


@app.route("/api/alerts/<int:alert_id>", methods=["DELETE"])
def delete_alert(alert_id):
    """Remove an alert from DB and delete its screenshot file."""
    photo = Photo.query.get_or_404(alert_id)
    # Delete physical file if it exists
    if os.path.exists(photo.filepath):
        os.remove(photo.filepath)
    db.session.delete(photo)
    db.session.commit()
    return jsonify({"deleted": alert_id}), 200


@app.route("/api/login", methods=["POST"])
def login():
    """Verify login credentials on the server securely."""
    from flask import request
    from werkzeug.security import check_password_hash
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400
        
    username = data.get('username')
    password = data.get('password')
    
    try:
        with open('hash.txt', 'r') as f:
            ADMIN_HASH = f.read().strip()
    except FileNotFoundError:
        return jsonify({"error": "Security configuration missing"}), 500
    
    if username == "admin" and check_password_hash(ADMIN_HASH, password):
        return jsonify({"message": "Access Granted", "token": "admin-session-ok"}), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401


@app.route("/api/stream")
def video_stream():
    from haris import generate_frames
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/api/stats")
def get_stats():
    """Return hourly alert counts for the AnalyticsChart."""
    photos = Photo.query.order_by(Photo.upload_time.asc()).all()
    valid = [p for p in photos if os.path.exists(p.filepath)]
    hourly = {}
    for p in valid:
        key = p.upload_time.strftime("%H:00")
        hourly[key] = hourly.get(key, 0) + 1
    return jsonify({
        "total": len(valid),
        "hourly": hourly
    })


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
