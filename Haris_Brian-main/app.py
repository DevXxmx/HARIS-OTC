from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
from flask import Flask, jsonify, send_from_directory
import os


app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///photos.db'


db = SQLAlchemy()
db.init_app(app)
CORS(app)


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


@app.route("/api/stream")
def video_stream():
    from flask import Response
    from haris import generate_frames
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
